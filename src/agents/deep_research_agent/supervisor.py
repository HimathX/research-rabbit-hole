"""Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting
"""

import asyncio
from typing import Literal

from langchain_core.messages import (
    HumanMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt, StreamWriter

from agents.deep_research_agent.prompts import (
    lead_researcher_prompt, 
    final_report_generation_prompt,
    clarify_with_user_instructions
)
from agents.deep_research_agent.research_agent import researcher_agent
from agents.deep_research_agent.state import (
    DeepResearchState, 
    ConductResearch, 
    ResearchComplete,
    ClarifyWithUser,
    DelegateToAnalyst
)
from agents.deep_research_agent.utils import get_today_str, think_tool
from agents.llama_guard import llama_guard_input  # SAFETY
from core import get_model, settings

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
max_researcher_iterations = 6
max_concurrent_researchers = 3

# ===== SUPERVISOR NODES =====

def get_notes_from_tool_calls(messages: list) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

async def scope_research(state: DeepResearchState) -> Command[Literal["supervisor"]]:
    """Analyze the user request and ask for clarification if needed (Human-in-the-Loop)."""
    messages = state.get("messages", [])
    
    # If we already have a research brief, skip scoping
    if state.get("research_brief"):
        return Command(goto="supervisor")

    # Use structured output to decide if clarification is needed
    scoping_model = supervisor_model.with_structured_output(ClarifyWithUser)
    
    prompt = clarify_with_user_instructions.format(
        messages=messages,
        date=get_today_str()
    )
    
    try:
        response: ClarifyWithUser = await scoping_model.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        # Fallback if structured output fails
        print(f"Scoping failed: {e}")
        return Command(goto="supervisor", update={"research_brief": messages[-1].content})

    if response.need_clarification:
        # INTERRUPT: Pause execution and ask user
        user_feedback = interrupt(response.question)
        
        # Add the interaction to history so the model knows clarity was provided
        new_messages = [
            HumanMessage(content=f"Clarification question: {response.question}"),
            HumanMessage(content=f"User Answer: {user_feedback}")
        ]
        
        # Recursively re-scope with new info
        return Command(
            goto="scope_research", 
            update={"messages": new_messages}
        )
    
    # Ready to proceed
    return Command(
        goto="supervisor", 
        update={"research_brief": response.verification} # Use the verification/summary as the brief
    )

async def supervisor(state: DeepResearchState, writer: StreamWriter = lambda _: None) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities."""
    supervisor_messages = state.get("messages", [])
    research_brief = state.get("research_brief", supervisor_messages[-1].content) # Fallback

    writer({"status": "Supervisor is planning research..."})

    # Prepare system message
    system_message = lead_researcher_prompt.format(
        date=get_today_str(), 
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations,
        research_brief=research_brief # Inject the scoped brief
    )
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
            from agents.data_analyst_agent import data_analyst_agent # Lazy import
            
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

# Nodes
supervisor_builder.add_node("llama_guard_input", llama_guard_input) # Safety
supervisor_builder.add_node("scope_research", scope_research)       # Interrupts
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("compile_report", compile_report)

# Edges
supervisor_builder.add_edge(START, "llama_guard_input")
supervisor_builder.add_edge("llama_guard_input", "scope_research")
# scope_research determines if it goes to interrupt or supervisor via Command
# supervisor determines if it goes to tools or report via Command
# supervisor_tools determines if it loops back or report via Command
# compile_report ends

deep_research_agent = supervisor_builder.compile()
