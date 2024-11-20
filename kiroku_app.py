# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

from agents.states import *
from copy import deepcopy
import streamlit as st
from IPython.display import display, Image
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langgraph.graph import StateGraph, START, END
import logging
import markdown
import os
import shutil
import subprocess
import re
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

st.set_page_config(page_title="Kiroku Document Writing App", layout="wide")

# Initialize session state variables
if 'kiroku_initialized' not in st.session_state:
    st.session_state.kiroku_initialized = False
    st.session_state.writer = None
    st.session_state.state_values = {}
    st.session_state.references = []
    st.session_state.next_state = -1
    st.session_state.filename = ""
    st.session_state.images = ""
    st.session_state.cache = set()
    st.session_state.draft = ""
    st.session_state.atlas_message = ""
    st.session_state.current_state = ""
    st.session_state.messages = []
    st.session_state.instruction = ""
    st.session_state.instruction_input_value = ''  # Initialize here

class DocumentWriter:
    def __init__(
            self,
            suggest_title=False,
            generate_citations=True,
            model_name="openai",
            temperature=0.0):
        self.suggest_title = suggest_title
        self.generate_citations = generate_citations
        self.state = None
        self.set_thread_id(1)
        models = {
            "openai": "gpt-4o-mini",
            "openai++": "gpt-4o"
        }
        assert model_name in ["openai", "openai++"]
        model = models.get(model_name, "openai")
        # if user did not specify the beefed up models upfront, we try to
        # use cheaper models whenever possible for simpler tasks.
        if model_name in ["openai", "openai++"]:
            self.model_m = ChatOpenAI(
                model=model, temperature=temperature)
            self.model_p = ChatOpenAI(
                model=models["openai++"], temperature=temperature)
        self.state_nodes = {
            node.name : node
            for node in [
                SuggestTitle(self.model_m),
                SuggestTitleReview(self.model_m),
                InternetSearch(self.model_p),
                TopicSentenceWriter(self.model_m),
                TopicSentenceManualReview(self.model_m),
                PaperWriter(self.model_p),
                WriterManualReviewer(self.model_m),
                ReflectionReviewer(self.model_p),
                ReflectionManualReview(self.model_m),
                WriteAbstract(self.model_p),
                GenerateReferences(self.model_m),
                GenerateCitations(self.model_m),
                GenerateFigureCaptions(self.model_m),
            ] if self.mask_nodes(node.name)
        }
        self.create_graph(suggest_title)

    def mask_nodes(self, name):
        '''
        We do not process nodes if user does not want to run that phase.
        :param name: name of the node.
        :return: True if we keep nodes, False otherwise
        '''
        if (
                name in ["suggest_title", "suggest_title_review"] and
                not self.suggest_title):
            return False
        if name in ["generate_references", "generate_citations"] and not self.generate_citations:
            return False
        return True

    def create_graph(self, suggest_title):
        '''
        Builds a graph to execute the different phases of a document writing.

        :param suggest_title: If we are to suggest a better title for the paper.
        :return: Nothing.
        '''
        memory = MemorySaver()

        builder = StateGraph(AgentState)

        # Add nodes to the graph
        for name, state in self.state_nodes.items():
            builder.add_node(name, state.run)

        # Add edges to the graph
        if suggest_title:
            builder.add_conditional_edges(
                "suggest_title_review",
                self.is_title_review_complete,
                {
                    "next_phase": "internet_search",
                    "review_more": "suggest_title"
                }
            )
        builder.add_conditional_edges(
            "topic_sentence_manual_review",
            self.is_plan_review_complete,
            {
                "topic_sentence_manual_review": "topic_sentence_manual_review",
                "paper_writer": "paper_writer"
            }
        )

        builder.add_conditional_edges(
            "writer_manual_reviewer",
            self.is_generate_review_complete,
            {
                "manual_review": "writer_manual_reviewer",
                "reflection": "reflection_reviewer",
                "finalize": "write_abstract"
            }
        )
        if suggest_title:
            builder.add_edge("suggest_title", "suggest_title_review")
        builder.add_edge("internet_search", "topic_sentence_writer")
        builder.add_edge("topic_sentence_writer", "topic_sentence_manual_review")
        builder.add_edge("paper_writer", "writer_manual_reviewer")
        builder.add_edge("reflection_reviewer", "additional_reflection_instructions")
        builder.add_edge("additional_reflection_instructions", "paper_writer")
        if self.generate_citations:
            builder.add_edge("write_abstract", "generate_references")
            builder.add_edge("generate_references", "generate_citations")
            builder.add_edge("generate_citations", "generate_figure_captions")
        else:
            builder.add_edge("write_abstract", "generate_figure_captions")
        builder.add_edge("generate_figure_captions", END)

        # Starting state is either suggest_title or planner.
        if suggest_title:
            builder.set_entry_point("suggest_title")
        else:
            builder.set_entry_point("internet_search")

        self.interrupt_after = []
        self.interrupt_before = [ "suggest_title_review" ] if suggest_title else []
        self.interrupt_before.extend([
            "topic_sentence_manual_review",
            "writer_manual_reviewer",
            "additional_reflection_instructions",
        ])
        if self.generate_citations:
            self.interrupt_before.append("generate_citations")
        # Build graph
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            debug=False
        )

    def is_title_review_complete(self, state: AgentState) -> str:
        '''
        Checks if title review is complete based on an END instruction.

        :param state: state of agent.
        :return: next state of agent.
        '''
        if not state["messages"]:
            return "next_phase"
        else:
            return "review_more"

    def is_plan_review_complete(self, state: AgentState, config: dict) -> str:
        '''
        Checks if plan manual review is complete based on an empty instruction.

        :param state: state of agent.
        :return: next state of agent.
        '''
        if config["configurable"]["instruction"]:
            return "topic_sentence_manual_review"
        else:
            return "paper_writer"

    def is_generate_review_complete(self, state: AgentState, config: dict) -> str:
        '''
        Checks if review of generation phase is complete based on number of revisions.

        :param state: state of agent.
        :return: next state to go.
        '''
        if config["configurable"]["instruction"]:
            return "manual_review"
        elif state["revision_number"] <= state["max_revisions"]:
            return "reflection"
        else:
            return "finalize"

    def invoke(self, state, config):
        '''
        Invokes the multi-agent system to write a paper.

        :param state: state of initial invokation.
        :return: draft
        '''
        config = { "configurable": config }
        config["configurable"]["thread_id"] = self.get_thread_id()
        response = self.graph.invoke(state, config)
        self.state = response
        draft = response.get("draft", "").strip()
        # we have to do this because the LLM sometimes decide to add
        # this to the final answer.
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return draft

    def stream(self, state, config):
        '''
        Invokes the multi-agent system to write a paper.

        :param state: state of initial invokation.
        :return: full state information
        '''
        config = { "configurable": config }
        config["configurable"]["thread_id"] = self.get_thread_id()
        for event in self.graph.stream(state, config, stream_mode="values"):
            pass
        draft = event["draft"]
        # we have to do this because the LLM sometimes decide to add
        # this to the final answer.
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return draft

    def get_state(self):
        """
        Returns the full state of the document writing process.
        :return: Generated state from invoke
        """
        config = { "configurable": { "thread_id": self.get_thread_id() }}
        return self.graph.get_state(config)

    def update_state(self, new_state):
        """
        Updates the state of langgraph.
        :param new_state:
        :return: None
        """
        config = { "configurable": { "thread_id": self.get_thread_id() }}
        self.graph.update_state(config, new_state.values)

    def get_thread_id(self):
        return str(self.thread_id)

    def set_thread_id(self, thread_id):
        self.thread_id = str(thread_id)

    def draw(self):
        display(
            Image(
                self.graph.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )

class KirokuUI:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.images_dir = os.path.join(self.working_dir, "images")
        self.filename = ""

        # Ensure images directory exists
        os.makedirs(self.images_dir, exist_ok=True)

    def read_initial_state(self, filepath):
        '''
        Reads initial state from a YAML file and initializes missing keys.
        '''
        try:
            with open(filepath, 'r') as file:
                state = yaml.load(file, Loader=yaml.Loader)
        except Exception as e:
            logger.error(f"Cannot load YAML file: {e}")
            st.error(f"Error loading YAML file: {e}")
            return {}

        # Ensure all required keys are initialized
        state.setdefault("sentences_per_paragraph", 4)
        state.setdefault("messages", [])  # Initialize messages as an empty list
        state.setdefault("review_topic_sentences", [])
        state.setdefault("review_instructions", [])
        state.setdefault("content", [])
        state.setdefault("references", [])
        state.setdefault("cache", set())
        state.setdefault("revision_number", 1)
        state.setdefault("plan", "")
        state.setdefault("critique", "")
        state.setdefault("task", "")

        # Extract and remove optional configurations
        st.session_state.suggest_title = state.pop("suggest_title", False)
        st.session_state.generate_citations = state.pop("generate_citations", False)
        st.session_state.model_name = state.pop("model_name", "openai++")
        st.session_state.temperature = state.pop("temperature", 0.0)

        # Combine hypothesis and instructions
        state["hypothesis"] = (
            state.get("hypothesis", "") + "\n\n" + state.pop("instructions", "")
        )
        return state

    def process_file(self, uploaded_file):
        '''
        Processes the uploaded YAML configuration file.
        '''
        try:
            # Save uploaded file to working directory
            filepath = os.path.join(self.working_dir, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            self.filename = filepath
            st.session_state.filename = self.filename
            logger.warning(f"Processing file: {self.filename}")

            # Read initial state
            state_values = self.read_initial_state(self.filename)
            if not state_values:
                st.error("Failed to read the YAML configuration.")
                return

            st.session_state.state_values = state_values

            # Initialize DocumentWriter
            st.session_state.writer = DocumentWriter(
                suggest_title=st.session_state.suggest_title,
                generate_citations=st.session_state.generate_citations,
                model_name=st.session_state.model_name,
                temperature=st.session_state.temperature
            )
            st.session_state.writer.set_thread_id(1)  # Set thread ID if needed
            st.session_state.kiroku_initialized = True
            st.success("YAML configuration loaded successfully.")

            # Initialize references and cache
            st.session_state.references = []
            st.session_state.cache = set()
            st.session_state.draft = ""
            st.session_state.atlas_message = ""
            st.session_state.current_state = ""
            st.session_state.messages = []
            st.session_state.instruction = ""

            # Start the initial step
            self.initial_step()

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            st.error(f"Error processing file: {e}")

    def initial_step(self):
        '''
        Performs the initial step of the document generation.
        '''
        try:
            # Ensure messages key exists in state_values
            st.session_state.state_values.setdefault("messages", [])

            # Invoke the writer with the initial state
            with st.spinner('Initializing document generation...'):
                draft = st.session_state.writer.invoke(st.session_state.state_values, {})
                st.session_state.draft = draft
                state = st.session_state.writer.get_state()
                st.session_state.current_state = state.values.get("state", "")
                try:
                    st.session_state.next_state = state.next[0]
                except (IndexError, AttributeError):
                    st.session_state.next_state = "NONE"

            # Generate atlas message
            st.session_state.atlas_message = self.atlas_message(st.session_state.next_state)

        except Exception as e:
            logger.error(f"Error in initial step: {e}")
            st.error(f"Error in initial step: {e}")

    def save_as(self):
        '''
        Saves the project in various formats.
        '''
        try:
            filename = st.session_state.filename
            state = st.session_state.writer.get_state()

            draft = state.values.get("draft", "").strip()
            draft = re.sub(r'\/?file=', '', draft)
            plan = state.values.get("plan", "")
            review_topic_sentences = "\n\n".join(state.values.get("review_topic_sentences", []))
            review_instructions = "\n\n".join(state.values.get("review_instructions", []))
            content = "\n\n".join(state.values.get("content", []))

            dir_name = os.path.splitext(filename)[0]
            dir_path = dir_name  # Save in the same directory
            try:
                shutil.rmtree(dir_path)
            except FileNotFoundError:
                pass
            os.makedirs(dir_path, exist_ok=True)

            # Symlink images
            images_dest = os.path.join(dir_path, "images")
            try:
                os.symlink(self.images_dir, images_dest)
            except FileExistsError:
                pass

            base_filename = os.path.join(dir_path, os.path.basename(dir_name))

            # Save Markdown
            with open(f"{base_filename}.md", "w", encoding='utf-8') as fp:
                fp.write(draft)
            logger.warning(f"Saved file {base_filename}.md")

            # Save HTML
            html = markdown.markdown(draft)
            with open(f"{base_filename}.html", "w", encoding='utf-8') as fp:
                fp.write(html)

            # Convert to DOCX using Pandoc
            try:
                subprocess.run(
                    [
                        "pandoc",
                        "-s", f"{base_filename}.html",
                        "-f", "html",
                        "-t", "docx",
                        "-o", f"{base_filename}.docx"
                    ],
                    check=True
                )
                logger.warning(f"Saved file {base_filename}.docx")
            except subprocess.CalledProcessError:
                logger.error("Failed to convert HTML to DOCX. Ensure Pandoc is installed.")
                st.error("Failed to convert HTML to DOCX. Ensure Pandoc is installed.")

            # Save additional files
            with open(f"{base_filename}_ts.txt", "w", encoding='utf-8') as fp:
                fp.write(review_topic_sentences)
            with open(f"{base_filename}_wi.txt", "w", encoding='utf-8') as fp:
                fp.write(review_instructions)
            with open(f"{base_filename}_plan.md", "w", encoding='utf-8') as fp:
                fp.write(plan)
            with open(f"{base_filename}_content.txt", "w", encoding='utf-8') as fp:
                fp.write(content)

            st.success(f"Project saved successfully in {dir_path}")

        except Exception as e:
            logger.error(f"Error saving project: {e}")
            st.error(f"Error saving project: {e}")

    def update_refs(self, selected_refs):
        '''
        Updates the references based on user selection.
        '''
        try:
            state = st.session_state.writer.get_state()
            references = selected_refs
            logger.warning("Keeping the following references")
            for ref in references:
                logger.warning(ref)
            state.values["references"] = '\n'.join(references)
            st.session_state.writer.update_state(state)
            st.session_state.references = references
            st.success("References updated successfully.")
            # Proceed to the next step after updating references
            self.update_instruction("")  # Empty instruction to move forward
        except Exception as e:
            logger.error(f"Error updating references: {e}")
            st.error(f"Error updating references: {e}")

    def atlas_message(self, state_name):
        '''
        Generates atlas messages based on the current state.
        '''
        messages = {
            "suggest_title_review":
                "Please suggest review instructions for the title.",
            "topic_sentence_manual_review":
                "Please suggest review instructions for the topic sentences.",
            "writer_manual_reviewer":
                "Please suggest review instructions for the main draft.",
            "additional_reflection_instructions":
                "Please provide additional instructions for the overall paper review.",
            "generate_citations":
                "Please look at the references tab and confirm the references."
        }

        instruction = messages.get(state_name, "")
        if instruction or state_name == "generate_citations":
            if state_name == "generate_citations":
                return instruction
            else:
                return instruction + " Type ENTER when done."
        else:
            return "We have reached the end."

    def create_ui(self):
        '''
        Renders the Streamlit UI components.
        '''
        tabs = st.tabs(["Initial Instructions", "Document Writing", "References"])

        # Tab 1: Initial Instructions
        with tabs[0]:
            st.header("Initial Instructions")
            uploaded_file = st.file_uploader("Upload YAML Configuration", type=["yaml"])
            if uploaded_file:
                self.process_file(uploaded_file)

            if st.session_state.kiroku_initialized:
                st.subheader("Initial State")
                st.json(st.session_state.state_values)

        # Tab 2: Document Writing
        with tabs[1]:
            st.header("Document Writing")
            if not st.session_state.kiroku_initialized:
                st.warning("Please upload the YAML configuration in the 'Initial Instructions' tab.")
                return

            # Display the current draft
            st.text_area("Echo", value=st.session_state.draft, height=300, key="echo_box", disabled=True)

            # Display atlas message
            if st.session_state.atlas_message:
                st.markdown(f"**System Message:** {st.session_state.atlas_message}")

            # Input for instructions
            instruction = st.text_input(
                "Instruction",
                placeholder="Enter your instruction here...",
                key="instruction_input",
                value=st.session_state.instruction_input_value
            )

            # Submit Instruction Button
            if st.button("Submit Instruction"):
                if instruction.strip() == "" and st.session_state.next_state not in ["generate_citations", "NONE", "END"]:
                    st.warning("Please enter a valid instruction.")
                else:
                    self.update_instruction(instruction)
                    # Clear the input field by resetting the value in session_state
                    st.session_state.instruction_input_value = ''
                    # Rerun the UI to update the input field
                    st.experimental_rerun()

            # Save button functionality
            if st.button("Save"):
                self.save_as()
                # Tab 3: References
        with tabs[2]:
            st.header("References")
            if not st.session_state.kiroku_initialized:
                st.warning("Please upload the YAML configuration in the 'Initial Instructions' tab.")
                return

            if st.session_state.generate_citations and st.session_state.references:
                st.subheader("Select References to Keep")
                # Use multiselect for better performance and usability
                selected_refs = st.multiselect(
                    "Select references to keep:",
                    options=st.session_state.references,
                    default=st.session_state.references
                )

    def update_instruction(self, instruction):
        '''
        Handles the instruction submission.
        '''
        try:
            # Update instruction in session state
            st.session_state.instruction = instruction

            with st.spinner('Processing...'):
                # Invoke the writer's update method
                if instruction.strip() != "":
                    # Non-empty instruction: proceed as normal
                    draft = st.session_state.writer.invoke(st.session_state.state_values, {"instruction": instruction})
                else:
                    # Empty instruction: proceed to next state without adding new instructions
                    draft = st.session_state.writer.invoke(st.session_state.state_values, {})

                st.session_state.draft = draft
                state = st.session_state.writer.get_state()
                st.session_state.current_state = state.values.get("state", "")

                try:
                    st.session_state.next_state = state.next[0]
                except (IndexError, AttributeError):
                    st.session_state.next_state = "NONE"

                # Handle specific state transitions
                if (
                    st.session_state.current_state == "reflection_reviewer" and
                    st.session_state.next_state == "additional_reflection_instructions"
                ):
                    st.session_state.draft = state.values.get("critique", "")

                if st.session_state.next_state == "generate_citations":
                    st.session_state.references = state.values.get("references", "").split('\n')

                if st.session_state.next_state in ["NONE", "END"]:
                    self.save_as()

                # Generate atlas message
                st.session_state.atlas_message = self.atlas_message(st.session_state.next_state)

            # No need to clear the instruction input here; it's handled in create_ui()

            # Rerun the UI to reflect updates
            st.rerun()

        except Exception as e:
            logger.error(f"Error updating instruction: {e}")
            st.error(f"Error updating instruction: {e}")

def run():
    working_dir = os.environ.get("KIROKU_PROJECT_DIRECTORY", os.getcwd())
    kiroku = KirokuUI(working_dir)
    kiroku.create_ui()

if __name__ == "__main__":
    n_errors = 0
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error("... We presently require an OPENAI_API_KEY.")
        n_errors += 1
    if not os.environ.get("TAVILY_API_KEY"):
        logging.error("... We presently require an TAVILY_API_KEY.")
        n_errors += 1
    if n_errors > 0:
        exit()

    run()
