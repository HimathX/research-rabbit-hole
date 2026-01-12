"""Deep Researcher Orchestrator.

This module provides the main 3-stage orchestrator for the deep research workflow:

    START → run_scoping → run_supervisor → compile_final_report → END

The orchestrator coordinates:
1. Scoping Phase: User clarification and research brief generation
2. Supervisor Phase: Multi-agent research coordination  
3. Report Phase: Final report compilation

This separation allows for:
- Clear separation of concerns between scoping and execution
- Easier testing and debugging of individual phases
- Flexibility to swap out or customize individual phases
"""

from typing import Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, StreamWriter

from agents.deep_research_agent.state import DeepResearchState
from agents.deep_research_agent.research_agent_scope import scope_research
from agents.deep_research_agent.supervisor import supervisor_graph


# ===== ORCHESTRATOR NODES =====

async def run_scoping(state: DeepResearchState, writer: StreamWriter = lambda _: None) -> dict:
    """
    Execute the scoping phase to clarify user intent and generate research brief.
    
    This node:
    1. Evaluates if clarification is needed (may interrupt for user input)
    2. Generates a detailed research brief with key areas and depth
    3. Passes the enriched state to the supervisor phase
    """
    writer({"status": "Analyzing research request..."})
    
    # Run the scoping subgraph
    result = await scope_research.ainvoke(state)
    
    writer({"status": f"Research brief generated. Depth: {result.get('brief_depth', 'moderate')}"})
    
    # Return the updated state fields from scoping
    return {
        "research_brief": result.get("research_brief", ""),
        "brief_key_areas": result.get("brief_key_areas", []),
        "brief_depth": result.get("brief_depth", "moderate"),
        "messages": result.get("messages", []),
    }


async def run_supervisor(state: DeepResearchState, writer: StreamWriter = lambda _: None) -> dict:
    """
    Execute the supervisor phase to coordinate multi-agent research.
    
    This node:
    1. Uses the research brief and key areas from scoping
    2. Delegates to specialized researcher agents
    3. Aggregates findings for final report generation
    """
    writer({"status": "Starting research coordination..."})
    
    # Log key areas being researched
    key_areas = state.get("brief_key_areas", [])
    if key_areas:
        writer({"status": f"Researching {len(key_areas)} key areas..."})
    
    # Run the supervisor subgraph
    result = await supervisor_graph.ainvoke(state)
    
    writer({"status": "Research phase complete."})
    
    # Return updated state from supervisor
    return {
        "messages": result.get("messages", []),
        "notes": result.get("notes", []),
        "raw_notes": result.get("raw_notes", []),
        "research_iterations": result.get("research_iterations", 0),
    }


async def compile_final_report(state: DeepResearchState, writer: StreamWriter = lambda _: None) -> dict:
    """
    Compile the final research report.
    
    Note: The actual report compilation happens in the supervisor graph's compile_report node.
    This node serves as a pass-through that could be extended for post-processing.
    """
    writer({"status": "Finalizing report..."})
    
    # The supervisor graph already handles report compilation
    # This node is available for any post-processing needs
    
    return {}


# ===== ROUTING LOGIC =====

def should_continue_to_supervisor(state: DeepResearchState) -> Literal["run_supervisor", "__end__"]:
    """
    Determine if we should proceed to the supervisor phase.
    
    Conditions to proceed:
    - We have a valid research brief
    
    Conditions to end early:
    - Scoping was interrupted for clarification (no research brief yet)
    """
    if state.get("research_brief"):
        return "run_supervisor"
    return "__end__"


# ===== GRAPH CONSTRUCTION =====

def build_deep_researcher_graph() -> StateGraph:
    """
    Build the 3-stage deep researcher orchestrator graph.
    
    Graph structure:
        START → run_scoping → run_supervisor → compile_final_report → END
                     ↓ (if interrupted)
                    END
    
    Returns:
        Compiled StateGraph for the deep researcher workflow
    """
    # Create the orchestrator graph
    orchestrator_builder = StateGraph(DeepResearchState)
    
    # Add nodes for each phase
    orchestrator_builder.add_node("run_scoping", run_scoping)
    orchestrator_builder.add_node("run_supervisor", run_supervisor)
    orchestrator_builder.add_node("compile_final_report", compile_final_report)
    
    # Add edges
    # START → run_scoping
    orchestrator_builder.add_edge(START, "run_scoping")
    
    # run_scoping → run_supervisor (if we have a brief) or END (if interrupted)
    orchestrator_builder.add_conditional_edges(
        "run_scoping",
        should_continue_to_supervisor,
        {
            "run_supervisor": "run_supervisor",
            "__end__": END,
        }
    )
    
    # run_supervisor → compile_final_report
    orchestrator_builder.add_edge("run_supervisor", "compile_final_report")
    
    # compile_final_report → END
    orchestrator_builder.add_edge("compile_final_report", END)
    
    return orchestrator_builder.compile()


# ===== COMPILED GRAPHS =====

# Main deep researcher graph with full 3-stage pipeline
deep_researcher = build_deep_researcher_graph()

# Export for backward compatibility
deep_research_agent = deep_researcher


# ===== UTILITY FUNCTIONS =====

async def run_deep_research(
    query: str,
    config: Optional[dict] = None,
) -> dict:
    """
    Convenience function to run deep research on a query.
    
    Args:
        query: The research query/topic from the user
        config: Optional LangGraph config (for checkpointing, callbacks, etc.)
    
    Returns:
        Final state with research results and report
    
    Example:
        result = await run_deep_research("Compare React vs Vue for large applications")
        print(result["messages"][-1].content)  # The final report
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
    }
    
    if config:
        return await deep_researcher.ainvoke(initial_state, config)
    return await deep_researcher.ainvoke(initial_state)
