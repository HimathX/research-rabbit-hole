"""Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

Note: The scoping phase (user clarification and research brief generation) has been
extracted to research_agent_scope.py and is orchestrated by deep_researcher.py.
"""

import asyncio
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, StreamWriter

from src.agents.deep_research_agent.prompts import (
    lead_researcher_prompt, 
    final_report_generation_prompt
)
from src.agents.deep_research_agent.research_agent import researcher_agent
from src.agents.deep_research_agent.state import (
    DeepResearchState, 
    ConductResearch, 
    ResearchComplete,
    DelegateToAnalyst
)
from src.agents.deep_research_agent.utils import get_today_str, think_tool
from src.core import get_model, settings

# Ensure async compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ===== CONFIGURATION =====

supervisor_tools = [ConductResearch, DelegateToAnalyst, ResearchComplete, think_tool]
supervisor_model = get_model(settings.DEFAULT_MODEL)
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# System constants
max_researcher_iterations = 10
max_concurrent_researchers = 3

# ===== SUPERVISOR NODES =====

def get_notes_from_tool_calls(messages: list) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


def get_depth_guidance(depth: str) -> str:
    """Generate research guidance based on depth setting."""
    depth_configs = {
        "shallow": "Focus on high-level overview only. Use 1-2 sub-agents maximum. Prioritize speed over comprehensiveness.",
        "moderate": "Balance depth and breadth. Use 2-3 sub-agents for distinct topics. Cover main points thoroughly.",
        "deep": "Conduct comprehensive investigation. Use up to max sub-agents. Explore all angles and gather extensive evidence.",
    }
    return depth_configs.get(depth, depth_configs["moderate"])


async def supervisor(state: DeepResearchState, writer: StreamWriter = lambda _: None) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities using the research brief and key areas from scoping phase."""
    supervisor_messages = state.get("messages", [])
    research_brief = state.get("research_brief", "")
    brief_key_areas = state.get("brief_key_areas", [])
    brief_depth = state.get("brief_depth", "moderate")
    
    # Fallback if no research brief (shouldn't happen with proper orchestration)
    if not research_brief and supervisor_messages:
        research_brief = supervisor_messages[-1].content

    writer({"status": "Supervisor is planning research..."})

    # Build enhanced system message with key areas and depth guidance
    key_areas_text = ""
    if brief_key_areas:
        key_areas_text = f"\n\n**Key Areas to Cover:**\n" + "\n".join(f"- {area}" for area in brief_key_areas)
    
    depth_guidance = get_depth_guidance(brief_depth)

    # Prepare system message
    system_message = lead_researcher_prompt.format(
        date=get_today_str(), 
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations,
        research_brief=research_brief + key_areas_text
    )
    
    # Add depth guidance to system message
    system_message += f"\n\n**Research Depth Guidance ({brief_depth}):** {depth_guidance}"
    
    # Ensure system message is first
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # Make decision
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: DeepResearchState, writer: StreamWriter = lambda _: None) -> Command[Literal["supervisor", "compile_report"]]:
    """Execute supervisor decisions."""
    supervisor_messages = state.get("messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    tool_messages = []
    all_raw_notes = []
    
    # Conditions
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        if exceeded_iterations:
             writer({"status": "Max iterations reached. Compiling report..."})
        elif research_complete:
             writer({"status": "Research complete. Compiling report..."})
             
        return Command(
            goto="compile_report",
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
            }
        )

    # Execute tools
    try:
        think_tool_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "think_tool"]
        conduct_research_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "ConductResearch"]

        # 1. Think Tool (Sync)
        for tool_call in think_tool_calls:
            observation = think_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=observation,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )

        # 2. Conduct Research (Async/Parallel)
        if conduct_research_calls:
            writer({"status": f"Delegating to {len(conduct_research_calls)} researchers..."})
            
            coros = [
                researcher_agent.ainvoke({
                    "messages": [HumanMessage(content=tool_call["args"]["research_topic"])],
                    "research_topic": tool_call["args"]["research_topic"]
                }) 
                for tool_call in conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*coros)

            for result, tool_call in zip(tool_results, conduct_research_calls):
                tool_messages.append(
                    ToolMessage(
                        content=result.get("compressed_research", "Error processing research"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )
                if "raw_notes" in result:
                    all_raw_notes.extend(result["raw_notes"])


        # 3. Delegate to Data Analyst (Async)
        delegate_analyst_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "DelegateToAnalyst"]
        if delegate_analyst_calls:
            from src.agents.data_analyst_agent import data_analyst_agent # Lazy import
            
            writer({"status": "Delegating to Data Analyst..."})
            coros = [
                data_analyst_agent.ainvoke({
                   "messages": [HumanMessage(content=tool_call["args"]["task_description"])]
                })
                for tool_call in delegate_analyst_calls
            ]
            
            analyst_results = await asyncio.gather(*coros)
            
            for result, tool_call in zip(analyst_results, delegate_analyst_calls):
                 # Extract the last AI message as the result
                 last_msg = result["messages"][-1]
                 tool_messages.append(
                    ToolMessage(
                        content=f"Data Analysis Result:\n{last_msg.content}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                 )

    except Exception as e:
        print(f"Error in supervisor_tools: {e}")
        return Command(goto="compile_report", update={})

    return Command(
        goto="supervisor",
        update={
            "messages": tool_messages,
            "raw_notes": all_raw_notes
        }
    )

async def compile_report(state: DeepResearchState, writer: StreamWriter = lambda _: None):
    """Compile final report from gathered notes."""
    writer({"status": "Writing final report..."})
    
    notes = state.get("notes", []) or state.get("raw_notes", [])
    findings = "\n\n".join(notes)
    
    prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", "Research task"),
        date=get_today_str(),
        findings=findings
    )
    
    response = await supervisor_model.ainvoke([HumanMessage(content=prompt)])
    writer({"status": "Report complete!"})
    return {"messages": [response]}


# ===== GRAPH CONSTRUCTION =====

supervisor_builder = StateGraph(DeepResearchState)

# Nodes (scope_research is now handled by research_agent_scope.py)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("compile_report", compile_report)

# Edges - Start directly with supervisor (scoping is handled externally)
supervisor_builder.add_edge(START, "supervisor")
# supervisor determines if it goes to tools or report via Command
# supervisor_tools determines if it loops back or report via Command
# compile_report ends

# This is the supervisor-only graph (expects research_brief to be pre-populated)
supervisor_graph = supervisor_builder.compile()

# Keep backward compatibility alias
supervisor_agent = supervisor_graph
