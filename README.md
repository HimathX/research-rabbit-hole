# 🔍 Research Rabbit Hole

> An intelligent deep research agent system built with LangGraph, FastAPI, and Streamlit that autonomously conducts comprehensive multi-agent research investigations.

A production-ready multi-agent research framework that combines user intent clarification, AI-powered research coordination, and intelligent report generation. Built with modern async Python, streaming APIs, and advanced LLM orchestration.

## 🎯 What This Does

The **Research Rabbit Hole** system helps you conduct deep, thorough research on any topic by:

1. **Understanding your intent** - Uses AI to clarify vague research requests and generate detailed research briefs
2. **Coordinating multi-agent research** - Deploys specialized research agents that work in parallel on different aspects of your topic
3. **Generating comprehensive reports** - Synthesizes findings into well-structured reports with citations and key insights

Perfect for market analysis, competitive intelligence, technical research, academic deep-dives, and any scenario requiring exhaustive information gathering.

### Key Capabilities

✨ **Multi-Agent Coordination** - Supervisor agent directs specialized researchers  
✨ **Intelligent Scoping** - Auto-clarifies ambiguous requests before research starts  
✨ **Real-time Streaming** - See research progress as it happens  
✨ **File & Web Access** - Research agents can read files and search the web  
✨ **Configurable Depth** - Control research breadth (shallow/moderate/deep)  
✨ **Thread-based History** - Maintain persistent research conversations  
✨ **LangSmith Integration** - Full observability and feedback tracking

## 🏗️ System Architecture

```
User Input (Streamlit UI)
         ↓
    [FastAPI Service]
         ↓
    ┌────────────────────────────────────────────┐
    │   Deep Research Agent (LangGraph)           │
    │  ┌──────────────────────────────────────┐   │
    │  │ 1️⃣  SCOPING PHASE                    │   │
    │  │ ├─ Intent Clarification              │   │
    │  │ │  (Ask clarifying questions)        │   │
    │  │ ├─ Research Brief Generation         │   │
    │  │ │  (Structured research plan)        │   │
    │  │ └─ Key Areas Extraction             │   │
    │  │    (Topics to cover)                 │   │
    │  └──────────────────────────────────────┘   │
    │                 ↓                            │
    │  ┌──────────────────────────────────────┐   │
    │  │ 2️⃣  RESEARCH PHASE                   │   │
    │  │ ├─ Supervisor Agent                  │   │
    │  │ │  (Coordinates research)            │   │
    │  │ ├─ Researcher Agents (Parallel)      │   │
    │  │ │  ├─ Web Search Researcher          │   │
    │  │ │  ├─ Document Analysis Researcher   │   │
    │  │ │  └─ Data Analyst                   │   │
    │  │ └─ Tools Available:                 │   │
    │  │    ├─ Web Search (DuckDuckGo)       │   │
    │  │    ├─ File Reading                   │   │
    │  │    └─ Calculator                     │   │
    │  └──────────────────────────────────────┘   │
    │                 ↓                            │
    │  ┌──────────────────────────────────────┐   │
    │  │ 3️⃣  REPORT PHASE                     │   │
    │  │ ├─ Synthesize Findings               │   │
    │  │ ├─ Format Report                     │   │
    │  │ └─ Extract Key Insights              │   │
    │  └──────────────────────────────────────┘   │
    └────────────────────────────────────────────┘
         ↓
    [Stream to UI]
         ↓
    User Feedback & Message History
```

### Data Flow

**Synchronous (Fast Response)**: User → Service → Agent → Final Report → UI  
**Streaming (Live Updates)**: User → Service → Agent → (token + message chunks) → UI  
**History**: Thread-based conversation storage with state persistence

## 🤖 Agents Overview

### Deep Research Agent (Primary)

The main agent that orchestrates the complete research workflow with three distinct phases:

#### Phase 1: Scoping (`research_agent_scope.py`)

Clarifies user intent and generates a structured research plan before expensive research begins.

| Component                    | Purpose                                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Intent Clarifier**         | Uses structured output to determine if user request is specific enough. If not, asks clarifying questions. |
| **Research Brief Generator** | Transforms conversation into detailed research brief with key areas to cover and desired depth.            |
| **State Updater**            | Enriches graph state with `research_brief`, `brief_key_areas`, `brief_depth` for next phase.               |

**Key Files**:

- `src/agents/deep_research_agent/research_agent_scope.py` - Scoping workflow
- `src/agents/deep_research_agent/prompts.py` - Prompts for clarification and brief generation

#### Phase 2: Research (`supervisor.py` + `research_agent.py`)

Multi-agent coordination where a supervisor delegates research to specialized agents.

| Component             | Purpose                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Supervisor Agent**  | Reads research brief, delegates tasks to researchers, manages iteration limits, and coordinates parallel research. |
| **Researcher Agents** | Specialized agents that execute research tasks using available tools. Run concurrently to maximize efficiency.     |
| **Tools**             | Web search (DuckDuckGo), file reading, calculator, thinking/reflection.                                            |

**Configuration**:

```python
max_concurrent_researchers = 3    # Max parallel research agents
max_researcher_iterations = 10    # Iteration limit per research session
```

**Key Files**:

- `src/agents/deep_research_agent/supervisor.py` - Supervisor orchestration logic
- `src/agents/deep_research_agent/research_agent.py` - Individual researcher agent
- `src/agents/tools.py` - Research tools (web search, file I/O, etc.)

#### Phase 3: Report Generation (`deep_researcher.py`)

Synthesizes research findings into a structured, readable report.

| Component              | Purpose                                                               |
| ---------------------- | --------------------------------------------------------------------- |
| **Report Compiler**    | Formats findings, extracts key insights, generates final report text. |
| **Message Aggregator** | Combines all research notes and outputs into coherent narrative.      |

**Key Files**:

- `src/agents/deep_research_agent/deep_researcher.py` - Phase orchestration
- `src/agents/deep_research_agent/state.py` - State schema and tool definitions

### State Management (`state.py`)

```python
@dataclass
class DeepResearchState:
    messages: list[ChatMessage]              # Conversation history
    research_brief: str                      # Generated research plan
    brief_key_areas: list[str]              # Topics to research
    brief_depth: str                         # shallow|moderate|deep
    notes: list[str]                         # Research findings
    # ... other fields
```

### Research Tools

Agents have access to:

- **Web Search** (`duckduckgo-search`) - Real-time internet search
- **File Reading** - Load and analyze documents
- **Calculator** - Numerical computations
- **Think** - Reflection tool for planning

## 🛠️ Technical Stack

| Layer             | Technology     | Purpose                                                  |
| ----------------- | -------------- | -------------------------------------------------------- |
| **Orchestration** | LangGraph v1.0 | Agent state machine, streaming, Command routing          |
| **LLMs**          | LangChain      | Multi-provider support (OpenAI, Anthropic, Google, etc.) |
| **Backend**       | FastAPI        | REST API with SSE streaming                              |
| **Frontend**      | Streamlit      | Web UI for chat and configuration                        |
| **Data**          | Pydantic       | Type-safe schemas and validation                         |
| **Storage**       | In-Memory      | Checkpoint storage for conversation state                |
| **Observability** | LangSmith      | Run tracing, feedback recording, debugging               |
| **Search**        | DuckDuckGo     | Web search for research                                  |

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- At least one LLM API key (OpenAI, Anthropic, etc.)

### Option 1: Local Python Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/research-rabbit-hole.git
cd research-rabbit-hole

# Install uv (recommended) or use pip
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync --frozen
source .venv/bin/activate

# Configure API key
echo 'OPENAI_API_KEY=your_key_here' >> .env

# Terminal 1: Start FastAPI service
python src/run_service.py

# Terminal 2: Start Streamlit app
streamlit run src/streamlit_app.py
```

The app opens at `http://localhost:8501`  
API available at `http://localhost:8080`

### Option 2: Docker Setup (Recommended)

```bash
git clone https://github.com/yourusername/research-rabbit-hole.git
cd research-rabbit-hole

# Configure
echo 'OPENAI_API_KEY=your_key_here' >> .env

# Launch with auto-reload
docker compose watch
```

Then navigate to `http://localhost:8501`

## 📂 Project Structure

```
research-rabbit-hole/
├── src/
│   ├── agents/                          # Agent implementations
│   │   ├── deep_research_agent/        # Main research agent (3-phase system)
│   │   │   ├── deep_researcher.py      # Phase orchestrator (START → scoping → research → report → END)
│   │   │   ├── research_agent_scope.py # Phase 1: Intent clarification & brief generation
│   │   │   ├── supervisor.py           # Phase 2: Multi-agent research coordinator
│   │   │   ├── research_agent.py       # Phase 2: Individual researcher agents
│   │   │   ├── prompts.py              # LLM system & user prompts
│   │   │   ├── state.py                # Graph state schema & tool definitions
│   │   │   └── utils.py                # Helper functions
│   │   ├── agents.py                   # Agent registry & loading
│   │   ├── tools.py                    # Shared research tools
│   │   └── lazy_agent.py               # Async agent loading
│   │
│   ├── service/                         # FastAPI backend
│   │   ├── service.py                  # Main service with /invoke & /stream endpoints
│   │   └── utils.py                    # Message conversion helpers
│   │
│   ├── client/                          # Client library
│   │   └── client.py                   # Async/sync client for service interaction
│   │
│   ├── core/                            # Core utilities
│   │   ├── llm.py                      # LLM provider initialization
│   │   └── settings.py                 # Configuration & environment variables
│   │
│   ├── schema/                          # Data models
│   │   ├── models.py                   # LLM model enums
│   │   └── schema.py                   # Chat messages, service schemas
│   │
│   ├── streamlit_app.py                 # Web UI (chat interface)
│   ├── run_service.py                   # Service entry point
│   └── run_agent.py                     # Direct agent invocation
│
├── docker/                              # Docker configurations
│   ├── Dockerfile.app                  # Streamlit app container
│   └── Dockerfile.service              # FastAPI service container
│
├── compose.yaml                         # Docker Compose (multi-service setup)
├── pyproject.toml                       # Dependencies & project metadata
├── .env.example                         # Environment variable template
└── tests/                               # Unit & integration tests
```

## 🔧 Configuration

### Environment Variables (`.env`)

**Required** (at least one LLM):

```bash
OPENAI_API_KEY=sk-...                   # OpenAI
# OR
ANTHROPIC_API_KEY=sk-ant-...            # Anthropic Claude
# OR
GROQ_API_KEY=...                        # Groq (with Llama models)
```

**Optional** (agent behavior):

```bash
DEFAULT_MODEL=gpt-4o                    # Default LLM to use
DEFAULT_AGENT=deep-research-agent       # Default agent

# Research depth: shallow, moderate, deep
RESEARCH_DEPTH=moderate

# Max concurrent researchers (1-5)
MAX_CONCURRENT_RESEARCHERS=3

# Max iterations per research session
MAX_RESEARCH_ITERATIONS=10
```

**Optional** (observability):

```bash
LANGSMITH_API_KEY=...                   # LangSmith tracing
LANGFUSE_TRACING=true                   # Langfuse observability
```

See [`.env.example`](./.env.example) for complete list.


## 📦 Dependencies

### Key Packages

- **langchain** - LLM abstractions & utilities
- **langgraph** - Agent orchestration & state management
- **fastapi** - REST API framework
- **streamlit** - Web UI
- **pydantic** - Data validation
- **duckduckgo-search** - Web search
- **langsmith** - Observability

See `pyproject.toml` for complete list with versions.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run: `pytest` and `pre-commit run --all-files`
5. Push and create a Pull Request

### Development Setup

```bash
uv sync --frozen
pre-commit install
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Credits

Built with:

- [LangChain](https://python.langchain.com/) - LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Streamlit](https://streamlit.io/) - Web UI
- [LangSmith](https://smith.langchain.com/) - Observability

