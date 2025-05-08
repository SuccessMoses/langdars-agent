from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


SUMMARIZER_SYSTEM = """
You are an expert technical analyst tasked with reviewing and summarizing the next steps taken to resolve a specific issue.
Your task involves analyzing the steps, identifying key issues, and suggesting improvements for better problem-solving.

Issue Description:
<issue_description>
{issue}
</issue_description>

Next Steps Taken:
<next_steps_taken>
{next_steps}
</next_steps_taken>
"""

SUMMARIZER_USER = """
Analyze the provided next steps to identify any potential issues, inefficiencies, or redundancies. Summarize what was attempted, assess the effectiveness of these steps, and suggest ways to improve the process.

Let's first answer few questions regarding next steps in short (if applicable):

1. Were the steps effective in reproducing the issue?
2. Was the issue localized correctly? Specify key files or components involved.
3. Were the required changes correctly identified and implemented?
4. Was the resolution verified properly to ensure the bug was fixed?
5. What errors occurred during bug fixing or reproduction?
6. Was the reproduction script accurate and complete? If not, what was missing?
7. What were the key learnings from these steps that could improve future processes?

Based on your analysis, suggest concrete improvements for the next steps. Ensure suggestions are actionable within the constraints of having access only to the repository (no internet or external documentation). Include specific file names or details where necessary.

Focus on clarity, conciseness, and actionable insights to improve the problem-solving process.
"""

EXPAND_SYSTEM = """
########## PLS FIX ME ####################
"""

summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARIZER_SYSTEM),
    ("user", SUMMARIZER_USER)
]).partial(candidate="")


expand_template = ChatPromptTemplate([
    ("system", EXPAND_SYSTEM),
    # Means the template will receive an optional list of messages under
    # the "conversation" key
    ("placeholder", "{conversation}")
    # Equivalently:
    # MessagesPlaceholder(variable_name="conversation", optional=True)
])


llm = ChatOllama(model="llama3", temperature=0,)

bound_llm = llm#.with_structured_output(GuessEquations)
summarizer = summarizer_prompt | bound_llm

expander = expand_template | llm

class DARS_CONFIG:
    summarize_expansion_context: bool = True
    n_lookahead: int = 5
    summarizer = summarizer
    model = llm
    expander = expander
    num_expansions: int
    issue: str