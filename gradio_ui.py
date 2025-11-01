import gradio as gr
from crewai import Agent, Task, Crew
# We use LangChain's ChatOpenAI wrapper for CrewAI compatibility
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Load OpenAI key from environment (ensure it's set in your .env file)
# The environment should handle the API key securely.
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
# Assuming OPENAI_API_KEY is available in the environment

def analyze_repo(repo_path: str):
    """
    Core function to run the CrewAI analysis on the provided repository path.
    """
    if not repo_path or repo_path.strip() == "":
        return "Please provide a valid repository path (e.g., local directory or URL) to begin the analysis."

    # Initialize the LLM. We'll use a fast, cost-effective model 
    # like gpt-3.5-turbo to simulate the performance of a "nano" model.
    try:
        llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.1, # Low temperature is better for code analysis
            # max_tokens=10000, # Optional: Adjust for longer reports
        )
    except Exception as e:
        return f"Error initializing LLM: {e}. Please ensure your OPENAI_API_KEY is correct."

    # Define the analyzer agent
    analyzer = Agent(
        role="Expert Python Code Reviewer",
        goal=f"Thoroughly analyze the Python code within the repository path '{repo_path}'.",
        backstory=(
            "You are an expert software engineer specializing in Databricks and Python best practices. "
            "Your task is to identify code smells, security risks, refactoring opportunities, and "
            "produce a single, high-quality, structured summary report."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # Define the task
    analyze_task = Task(
        description=f"Analyze the repository at '{repo_path}'. Create a structured summary that includes:"
                    "\n1. A high-level assessment of code quality."
                    "\n2. Specific refactoring suggestions (e.g., for readability, performance)."
                    "\n3. Identification of potential Databricks/Spark optimization issues."
                    "\n4. A final, single-block code quality score (e.g., A+, B-).",
        expected_output="A single, detailed, and professional structured markdown report containing all requested sections.",
        agent=analyzer,
        # Pass the repo_path via the description now that we are using Gradio as the entry point
    )

    # Instantiate the crew
    crew = Crew(
        agents=[analyzer],
        tasks=[analyze_task],
        verbose=True,
    )

    # Run the analysis
    try:
        # Note: The input is already included in the Task description
        result = crew.kickoff()
        return result
    except Exception as e:
        # Provide a user-friendly error message
        return f"An error occurred during the CrewAI analysis process: {e}"

# --- Gradio Interface Definition ---

# Input component for the repository path
repo_input = gr.Textbox(
    label="Repository Path or URL",
    placeholder="Enter the local path (e.g., /path/to/my/project) or a public URL (if supported by your tools).",
    lines=1,
    scale=3,
)

# Output component for the detailed report
analysis_output = gr.Markdown(
    label="Code Review Report (GPT-3.5-Turbo Analysis)",
    value="Report will appear here...",
)

# The main Gradio interface
iface = gr.Interface(
    fn=analyze_repo,
    inputs=repo_input,
    outputs=analysis_output,
    title="CrewAI Repository Code Reviewer",
    description=(
        "Welcome to the Code Reviewer powered by CrewAI. Enter your repository path below, "
        "and a specialized agent will perform a deep code quality analysis."
        "This system uses a fast LLM to ensure a quick and cost-effective review."
    ),
    live=False,
    flagging_mode='never', # Updated from allow_flagging='never'
    theme=gr.themes.Soft(),
)

# Launch the Gradio app (The environment handles the launch command)
if __name__ == "__main__":
    iface.launch()
