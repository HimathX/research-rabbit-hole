"""Research Agent Implementation.

This module implements a research agent that can perform iterative searches (web, RAG, files)
and synthesis to answer complex research questions.
"""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langgraph.graph import StateGraph, START, END

from agents.deep_research_agent.state import ResearcherState, ResearcherOutputState
from agents.deep_research_agent.utils import research_tools, get_today_str
from agents.deep_research_agent.prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message
from core import get_model, settings

# ===== CONFIGURATION =====

# Set up tools and model binding
tools = research_tools
tools_by_name = {tool.name: tool for tool in tools}

# Single base model using project settings
model = get_model(settings.DEFAULT_MODEL)
model_with_tools = model.bind_tools(tools)

# Reuse base model for summarization
summarization_model = model

# Separate instance for compression (using same model for now, but enabling larger context if needed)
compress_model = get_model(settings.DEFAULT_MODEL) 

# ===== AGENT NODES =====

from langgraph.types import StreamWriter

def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions."""
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt.format(date=get_today_str()))] + state["messages"]
            )
        ]
    }

def tool_node(state: ResearcherState, writer: StreamWriter = lambda _: None):
    """Execute all tool calls from the previous LLM response."""
    tool_calls = state["messages"][-1].tool_calls

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        
        # Streaming update
        if tool_name == "tavily_search":
             writer({"status": f"Searching web for: {tool_call['args'].get('query', '...')}"})
        elif tool_name == "read_local_file":
             writer({"status": f"Reading file: {tool_call['args'].get('file_path', '...')}"})
        elif tool_name == "database_search":
             writer({"status": f"Querying RAG: {tool_call['args'].get('query', '...')}"})
        
        tool = tools_by_name[tool_name]
        try:
             observations.append(tool.invoke(tool_call["args"]))
        except Exception as e:
             observations.append(f"Tool error: {e}")

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=str(observation),
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary."""

    system_message = compress_research_system_prompt.format(date=get_today_str())
    research_topic = state.get("research_topic", "General Research")
    
    messages = [SystemMessage(content=system_message)] + state.get("messages", []) + [HumanMessage(content=compress_research_human_message.format(research_topic=research_topic))]
    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
        "messages": [response] # Append final summary to messages if needed provided it matches schema
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()
