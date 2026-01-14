"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from src.agents.deep_research_agent.prompts import (
    clarify_with_user_instructions, 
    transform_messages_into_research_topic_prompt
)
from src.agents.deep_research_agent.state import (
    DeepResearchState,
    ClarifyWithUser, 
    ResearchBrief
)
from src.agents.deep_research_agent.utils import get_today_str
from src.core import get_model, settings

# ===== CONFIGURATION =====

# Initialize scoping model
scoping_model = get_model(settings.DEFAULT_MODEL)


# ===== WORKFLOW NODES =====

def clarify_with_user(state: DeepResearchState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    messages = state.get("messages", [])
    
    # If we already have a research brief, skip to write_research_brief
    if state.get("research_brief"):
        return Command(goto="write_research_brief")
    
    # Set up structured output model
    structured_output_model = scoping_model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=messages), 
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        # End the graph with an AI message asking for clarification
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=response.question)]},
        )

    # Proceed to generate research brief
    return Command(
        goto="write_research_brief", 
        update={"messages": [AIMessage(content=response.verification)]}
    )


async def clarify_with_user_async(state: DeepResearchState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Async version of clarify_with_user for better performance.
    """
    messages = state.get("messages", [])
    
    # If we already have a research brief, skip to write_research_brief
    if state.get("research_brief"):
        return Command(goto="write_research_brief")
    
    # Set up structured output model
    structured_output_model = scoping_model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = await structured_output_model.ainvoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=messages), 
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        # End the graph with an AI message asking for clarification
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=response.question)]},
        )

    # Proceed to generate research brief
    return Command(
        goto="write_research_brief", 
        update={"messages": [AIMessage(content=response.verification)]}
    )


def write_research_brief(state: DeepResearchState) -> dict:
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research, including:
    - Detailed research brief text
    - Key areas to cover (main topics)
    - Research depth (shallow/moderate/deep)
    """
    messages = state.get("messages", [])
    
    # Set up structured output model
    structured_output_model = scoping_model.with_structured_output(ResearchBrief)

    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(messages),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and metadata
    return {
        "research_brief": response.research_brief,
        "brief_key_areas": response.key_areas,
        "brief_depth": response.research_depth,
    }


async def write_research_brief_async(state: DeepResearchState) -> dict:
    """
    Async version of write_research_brief for better performance.
    """
    messages = state.get("messages", [])
    
    # Set up structured output model
    structured_output_model = scoping_model.with_structured_output(ResearchBrief)

    # Generate research brief from conversation history
    response = await structured_output_model.ainvoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(messages),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and metadata
    return {
        "research_brief": response.research_brief,
        "brief_key_areas": response.key_areas,
        "brief_depth": response.research_depth,
    }


# ===== GRAPH CONSTRUCTION =====

def build_scoping_graph(use_async: bool = True) -> StateGraph:
    """Build the scoping workflow graph.
    
    Args:
        use_async: Whether to use async node implementations (default True)
    
    Returns:
        Compiled scoping workflow graph
    """
    # Build the scoping workflow
    scoping_builder = StateGraph(DeepResearchState)

    # Add workflow nodes (use async versions by default for better performance)
    if use_async:
        scoping_builder.add_node("clarify_with_user", clarify_with_user_async)
        scoping_builder.add_node("write_research_brief", write_research_brief_async)
    else:
        scoping_builder.add_node("clarify_with_user", clarify_with_user)
        scoping_builder.add_node("write_research_brief", write_research_brief)

    # Add workflow edges
    scoping_builder.add_edge(START, "clarify_with_user")
    scoping_builder.add_edge("write_research_brief", END)

    return scoping_builder.compile()


# Compile default scoping graph (async)
scope_research = build_scoping_graph(use_async=True)
