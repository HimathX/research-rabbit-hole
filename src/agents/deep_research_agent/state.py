"""State Definitions for Deep Research Agent.

This module defines the state objects and tools used for the deep research workflow.
"""

import operator
from typing import Annotated, List, TypedDict, Sequence, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE DEFINITIONS =====

class DeepResearchState(MessagesState):
    """
    Main state for the deep research supervisor.
    """
    # Messages are handled by MessagesState ('messages' key)
    
    # Detailed research brief that guides the overall research direction
    research_brief: str = ""
    # Key areas to cover during research (from scoping phase)
    brief_key_areas: list[str] = []
    # Research depth: shallow, moderate, or deep (from scoping phase)
    brief_depth: str = "moderate"
    # Processed and structured notes ready for final report generation
    notes: Annotated[list[str], operator.add] = []
    # Counter tracking the number of research iterations performed
    research_iterations: int = 0
    # Raw unprocessed research notes collected from sub-agent research
    raw_notes: Annotated[list[str], operator.add] = []


class ResearcherState(MessagesState):
    """
    State for the individual research sub-agent.
    """
    # Messages are handled by MessagesState ('messages' key)
    
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str = ""
    raw_notes: Annotated[List[str], operator.add] = []


class ResearcherOutputState(TypedDict):
    """Output state for the research agent."""
    compressed_research: str
    raw_notes: List[str]
    messages: List[BaseMessage]

# ===== TOOL SCHEMAS =====

class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class DelegateToAnalyst(BaseModel):
    """Tool for delegating data analysis or code execution tasks to a Data Analyst agent."""
    task_description: str = Field(
        description="Detailed description of the analysis or calculation to perform. Can include data snippets or requests for plots.",
    )

class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
    pass

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions during scoping phase."""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for research brief generation."""
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class ResearchBrief(BaseModel):
    """Enhanced schema for research brief generation with structured metadata.
    
    This model captures not just the research question, but also the key areas
    to investigate and the appropriate depth of research needed.
    """
    research_brief: str = Field(
        description="A detailed research brief that will guide the research process. "
                    "Should include context, specific questions to answer, and any constraints.",
    )
    key_areas: list[str] = Field(
        default_factory=list,
        description="Main topics or areas to cover during research. "
                    "These become the focus areas for sub-agent delegation.",
    )
    research_depth: str = Field(
        default="moderate",
        description="The depth of research required: 'shallow' for quick overviews, "
                    "'moderate' for balanced research, 'deep' for comprehensive investigation.",
    )
