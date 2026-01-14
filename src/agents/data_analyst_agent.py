"""Data Analyst Agent.

This agent specializes in analyzing data, running python code, and generating visualizations.
It uses a secure Python REPL environment to execute code.
"""

from typing import Annotated, Literal

from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import RetryPolicy

from src.core import get_model, settings

# ===== CONFIGURATION =====

# Initialize Python REPL
repl = PythonREPL()

@tool
def python_interpreter(code: str) -> str:
    """Execute Python code and return the output. 
    
    Use this to perform calculations, data analysis, or generate plots.
    If you make plots, save them as files (e.g., 'plot.png') in the current directory.
    The output should be printed to stdout.
    """
    try:
        return repl.run(code)
    except Exception as e:
        return f"Error executing code: {e}"

tools = [python_interpreter]
model = get_model(settings.DEFAULT_MODEL).bind_tools(tools)

# ===== AGENT NODES =====

class DataAnalystState(MessagesState):
    """State for the data analyst."""
    pass

def llm_call(state: DataAnalystState):
    """Decide on code to run."""
    system_message = SystemMessage(content="""You are an expert Data Analyst and Python Programmer.
Your goal is to answer the user's questions by writing and executing Python code.

You have access to a `python_interpreter` tool.
- Use pandas for data manipulation.
- Use matplotlib/seaborn for plotting.
- ALWAYS print the final result of your calculation so it can be captured.
- If asked to create a plot, save it to the current directory with a descriptive name.

When you have the answer, respond with the final answer in natural language, citing the results from your code.
""")
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}

def tool_node(state: DataAnalystState):
    """Execute python code."""
    last_message = state["messages"][-1]
    
    tools_by_name = {t.name: t for t in tools}
    outputs = []
    
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        try:
            output = tool.invoke(tool_call["args"])
        except Exception as e:
            output = f"Execution error: {str(e)}"
            
        outputs.append(
            ToolMessage(
                content=str(output),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
        )
        
    return {"messages": outputs}

def should_continue(state: DataAnalystState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "__end__"

# ===== GRAPH CONSTRUCTION =====

builder = StateGraph(DataAnalystState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "__end__": END
    }
)
builder.add_edge("tool_node", "llm_call")

data_analyst_agent = builder.compile()
