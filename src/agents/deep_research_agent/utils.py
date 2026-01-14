"""Research Utilities and Tools.

This module provides search, content processing, and file system utilities for the research agent.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient

from src.agents.tools import database_search
from src.core import get_model, settings

# Use local imports for schemas to avoid circular dependency issues if they arise, 
# but here we'll assume they will be in state.py or prompts.py
# For now, defining Summary schema here if it's not easily imported or importing from state if I create it first.
# actually, I'll create the State file next, so I will define a local Summary model for now or import it from state later.
# To be safe and clean, I will define the Pydantic models needed for structured output right here or expect them in state.
# Let's import from .state once we create it. For now I'll define it strictly here to avoid import error before state is created.
from pydantic import BaseModel, Field

class Summary(BaseModel):
    """Schema for webpage content summarization."""
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")

# Load environment variables
load_dotenv()

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    today = datetime.now()
    day = today.day
    return today.strftime(f"%a %b {day}, %Y")

def get_current_dir() -> Path:
    """Get the current directory."""
    return Path.cwd()

# ===== CONFIGURATION =====

# Use default model for summarization
summarization_model = get_model(settings.DEFAULT_MODEL)
tavily_client = TavilyClient()

# ===== SEARCH FUNCTIONS =====

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> List[dict]:
    """Perform search using Tavily API for multiple queries."""
    search_docs = []
    for query in search_queries:
        try:
            result = tavily_client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic
            )
            search_docs.append(result)
        except Exception as e:
            print(f"Error searching for {query}: {e}")
            continue

    return search_docs

def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model."""
    from src.agents.deep_research_agent.prompts import summarize_webpage_prompt # Deferred import

    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)

        # Generate summary
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content
            ))
        ])

        # Format summary with clear structure
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL."""
    unique_results = {}
    for response in search_results:
        if not response or 'results' not in response:
            continue
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available."""
    summarized_results = {}
    for url, result in unique_results.items():
        if not result.get("raw_content"):
            content = result['content']
        else:
            content = summarize_webpage_content(result['raw_content'])

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }
    return summarized_results

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output."""
    if not summarized_results:
        return "No valid search results found."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# ===== TOOLS =====

@tool
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return (default: 3)
        topic: Topic to filter results by (default: "general")

    Returns:
        Formatted string of search results with summaries
    """
    search_results = tavily_search_multiple(
        [query],
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )
    unique_results = deduplicate_search_results(search_results)
    summarized_results = process_search_results(unique_results)
    return format_search_output(summarized_results)

@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    
    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded
    """
    return f"Reflection recorded: {reflection}"

@tool
def list_local_files(directory_path: str = ".") -> str:
    """List files in a local directory.
    
    Args:
        directory_path: The relative path to listing files from. Defaults to current directory ".".
    
    Returns:
        A formatted list of files and directories.
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"Error: Directory '{directory_path}' does not exist."
        
        items = []
        for p in path.iterdir():
            # Skip hidden files/dirs
            if p.name.startswith('.'):
                continue
            type_str = "DIR" if p.is_dir() else "FILE"
            items.append(f"[{type_str}] {p.name}")
            
        return "\n".join(items) if items else "Directory is empty."
    except Exception as e:
        return f"Error listing files: {e}"

@tool
def read_local_file(file_path: str) -> str:
    """Read the contents of a local file.
    
    Args:
        file_path: The path to the file to read.
        
    Returns:
        The content of the file or error message.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
            
        # Basic binary check extension/content could be added here
        return path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return f"Error reading file: {e}"

# Export tools
research_tools = [tavily_search, think_tool, database_search, list_local_files, read_local_file]
