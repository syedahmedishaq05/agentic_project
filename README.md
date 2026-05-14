#  AMARA - Multi-Agent Research Assistant

> An advanced AI-powered research assistant that synthesizes academic literature through a coordinated multi-agent pipeline, leveraging Retrieval-Augmented Generation (RAG) and large language models to answer complex research questions comprehensively.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Agent Descriptions](#agent-descriptions)
- [RAG Pipeline](#rag-pipeline)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Project Structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)

---

##  Overview

**AMARA** (Multi-Agent Research Assistant) is a sophisticated AI system designed to answer complex research questions by:

1. **Searching** academic databases (ArXiv, Semantic Scholar) for relevant papers
2. **Summarizing** key findings from multiple sources
3. **Critically analyzing** the literature for gaps and weaknesses
4. **Synthesizing** a comprehensive, well-cited answer grounded in academic literature

The system uses a **LangGraph-based orchestration pipeline** with four specialized agents working in sequence, complemented by **Retrieval-Augmented Generation (RAG)** to ensure responses are grounded in retrievable, verifiable chunks of text.

### Key Differentiators

- **Multi-agent workflow**: Each agent specializes in a specific task (search → summarize → critique → write)
- **Academic rigor**: Designed specifically for research synthesis with proper citations and critical analysis
- **RAG-grounded**: All outputs are grounded in retrieved chunks, reducing hallucinations
- **Flexible LLM support**: Works with Groq llama-3.3-70b-versatile, OpenAI GPT-4, or Anthropic Claude
- **Web UI**: User-friendly Streamlit interface for interactive research
- **Benchmarking**: Built-in evaluation framework comparing multi-agent vs. single-LLM performance

---

##  Features

### Core Capabilities

- **Intelligent Paper Retrieval**: Automatically extracts optimal keywords and searches ArXiv and Semantic Scholar
- **Structured Summarization**: Generates academic summaries covering core contributions, methodology, results, and relevance
- **Critical Analysis**: Identifies research gaps, methodological weaknesses, missing perspectives, and contradictions
- **Synthesis Writing**: Produces comprehensive, well-cited academic answers in APA format
- **RAG Integration**: Builds a FAISS vector index for semantic similarity search over paper contents
- **State Management**: LangGraph-based state orchestration with clean error handling
- **Performance Evaluation**: ROUGE and BLEU metrics comparing multi-agent vs. single-LLM baselines
- **Interactive UI**: Streamlit web interface for conversational interaction
- **Configurable**: Support for multiple LLM providers and customizable search parameters

### Quality Features

- Deduplication of search results across multiple sources
- Structured output with clear separation of concerns
- Graceful error handling with informative fallbacks
- Rate limiting for external API calls
- Caching and persistence of vector indices

---

##  Architecture

### System Overview

```
User Query
    ↓
[Searcher Agent] → Extract keywords, search ArXiv + Semantic Scholar
    ↓
[RAG Pipeline] → Build vector index from papers, retrieve relevant chunks
    ↓
[Summarizer Agent] → Generate structured summaries of papers
    ↓
[Critic Agent] → Analyze literature for gaps, weaknesses, consensus
    ↓
[Writer Agent] → Synthesize final answer with citations
    ↓
Final Report (with References)
```

### Agent Interactions

The agents operate through **LangGraph's StateGraph**, which manages:
- **State Schema** (`AMARState`): Passed through the entire pipeline
- **Node Functions**: Each agent reads state and returns updated state
- **Sequential Execution**: Ensures proper data flow and dependency management
- **Error Handling**: Captures exceptions and continues gracefully

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | State graph pipeline coordination |
| **LLMs** | Groq llama-3.3-70b-versatile / OpenAI GPT-4 / Anthropic Claude | Language understanding and generation |
| **Search** | ArXiv API, Semantic Scholar API | Academic paper retrieval |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Dense vector representations |
| **Vector DB** | FAISS | Similarity search over paper chunks |
| **Web UI** | Streamlit | Interactive user interface |
| **Evaluation** | ROUGE, BLEU, NLTK | Response quality metrics |
| **Framework** | LangChain | LLM interactions and chains |

---

##  Installation

### Prerequisites

- Python 3.9+
- pip or conda
- API keys for at least one LLM provider:
  - **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - **Anthropic**: [https://console.anthropic.com/](https://console.anthropic.com/)

### Step 1: Clone Repository

```bash
git clone https://github.com/syedahmedishaq05/agentic_project.git
cd agentic_project
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **langchain** & **langgraph**: Agent orchestration
- **groq**, **openai** & **anthropic**: LLM APIs
- **sentence-transformers**: Embedding models
- **faiss-cpu**: Vector similarity search
- **streamlit**: Web interface
- **arxiv**: ArXiv paper search
- **rouge-score** & **nltk**: Evaluation metrics
- And more (see `requirements.txt`)

### Step 4: Configure Environment

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```bash
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=groq  # or "openai" or "anthropic"
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

Optional but recommended:
```bash
SEMANTIC_SCHOLAR_API_KEY=...  # Get free at semanticscholar.org
MAX_PAPERS=10
MAX_CHUNKS=5
```

---

##  Quick Start

### Option 1: Interactive Web UI

```bash
streamlit run app.py
```

Opens a Streamlit app at `http://localhost:8501` where you can:
- Enter research questions interactively
- See live agent progress and state updates
- View final reports with citations
- Monitor performance metrics

### Option 2: Command-Line Usage

```python
from pipeline import build_graph

# Build the orchestration graph
graph = build_graph()
compiled_graph = graph.compile()

# Run the pipeline
final_state = compiled_graph.invoke({
    "query": "What are the latest advances in multi-agent AI systems?"
})

print(final_state["final_answer"])
print(final_state["references"])
```

### Option 3: Evaluate Performance

```bash
python evaluator.py
```

Runs the evaluation framework on benchmark queries, comparing AMARA vs. single-LLM baseline:

```python
from evaluator import EvaluationRunner

runner = EvaluationRunner()
result = runner.evaluate_single_query(
    "How do vision transformers work?",
    ground_truth="..."  # optional
)

print(f"AMARA ROUGE-1: {result['amara_metrics']['rouge1_f']}")
print(f"Baseline ROUGE-1: {result['baseline_metrics']['rouge1_f']}")
```

---

##  Configuration

All configuration is managed in `config.py` and `.env`:

### LLM Settings

```python
LLM_PROVIDER = "groq"  # or "openai" or "anthropic"
GROQ_API_KEY = "your_groq_api_key_here"
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
GROQ_MODEL = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
```

Use `get_llm(temperature=0.3)` to instantiate a configured LLM with custom temperature.

### RAG Settings

```python
VECTOR_STORE_PATH = "./data/faiss_index"  # Where to persist embeddings
MAX_PAPERS = 10  # Max papers retrieved per query
MAX_CHUNKS = 5   # Max chunks returned per RAG query
```

### Search Settings

```python
SEMANTIC_SCHOLAR_API_KEY = ""  # Get free at semanticscholar.org
```

---

##  Usage Guide

### Basic Query Flow

#### 1. **Searcher Agent**
Extracts 3-5 optimal keywords and retrieves papers from:
- **ArXiv** (via `arxiv` library)
- **Semantic Scholar** (via HTTP API)

Input: User's research question
Output: List of papers with title, authors, abstract, year, URL

#### 2. **RAG Pipeline**
Builds a searchable index of paper chunks:
- Chunks paper abstracts into overlapping segments
- Embeds chunks using SentenceTransformers
- Stores embeddings in FAISS index (or numpy fallback)
- Retrieves contextually relevant chunks for the query

#### 3. **Summarizer Agent**
Generates structured summaries for each paper:
- Core contribution (1-2 sentences)
- Methodology (1-2 sentences)
- Key results (1-2 sentences)
- Relevance to query (1 sentence)

#### 4. **Critic Agent**
Analyzes the literature holistically:
- **Research gaps**: Unanswered questions
- **Methodological weaknesses**: Common flaws
- **Missing perspectives**: Underrepresented viewpoints
- **Contradictions**: Disagreements across papers
- **Strengths**: Well-established findings
- **Consensus**: Most agreed-upon findings and contested points

#### 5. **Writer Agent**
Synthesizes all outputs into an academic report:
- **Introduction**: Context and significance
- **Literature review**: Synthesized findings with citations
- **Critical analysis**: Gaps and weaknesses integrated
- **Conclusion**: Direct evidence-based answer
- **References**: APA-formatted bibliography

### Example: Research Synthesis

```python
query = "How do large language models learn from human feedback?"

# Run full pipeline
result = pipeline.run(query)

# Access results
print("Keywords:", result["keywords"])
print("Papers retrieved:", len(result["papers"]))
print("Critical analysis:", result["critique"])
print("Final answer:", result["final_answer"])
print("References:", result["references"])
```

---

##  Agent Descriptions

### Searcher Agent (`searcher_agent.py`)

**Purpose**: Retrieve relevant academic papers

**Key Methods**:
- `extract_keywords(query)`: Uses LLM to extract 3-5 optimal search keywords
- `search_arxiv(keywords)`: Queries ArXiv API, returns 10 papers max
- `search_semantic_scholar(keywords)`: Queries Semantic Scholar API with rate limiting
- `run(query)`: Orchestrates full search pipeline

**Output State**:
```python
{
    "query": str,
    "keywords": List[str],
    "papers": List[Dict]  # [{title, authors, abstract, url, year, source, id}]
}
```

### Summarizer Agent (`summarizer_agent.py`)

**Purpose**: Generate structured summaries of papers

**Key Methods**:
- `summarize_paper(paper, query)`: LLM-powered summary of a single paper
- `run(searcher_output)`: Summarizes all papers in sequence

**Output**: Papers with added `summary` field containing structured text

**Summary Structure**:
```
1. Core Contribution (1-2 sentences)
2. Methodology (1-2 sentences)
3. Key Results (1-2 sentences)
4. Relevance (1 sentence)
```

### Critic Agent (`critic_agent.py`)

**Purpose**: Critical analysis of literature corpus

**Key Methods**:
- `critique_literature(query, papers)`: Identifies gaps, weaknesses, perspectives, contradictions
- `identify_consensus(query, papers)`: Finds agreed-upon findings and contested points
- `run(summarizer_output)`: Orchestrates full analysis

**Output State**:
```python
{
    "critique": str,      # Gap analysis, weaknesses, perspectives
    "consensus": str      # Agreed findings, contested points, maturity level
}
```

### Writer Agent (`writer_agent.py`)

**Purpose**: Synthesize final academic report

**Key Methods**:
- `format_references(papers)`: Converts to APA format
- `build_context(query, papers, critique, consensus)`: Assembles context for LLM
- `run(critic_output)`: Generates final answer

**Output State**:
```python
{
    "final_answer": str,  # 600-900 word academic synthesis
    "references": str     # APA-formatted bibliography
}
```

**Final Answer Structure**:
```
## Introduction
Brief context and significance

## Literature Review
Synthesized findings with inline citations (Author et al., Year)

## Critical Analysis
Integrated gaps and weaknesses

## Conclusion
Direct evidence-based answer

## References
[1] Author et al. (Year). Title. Retrieved from URL
...
```

---

##  RAG Pipeline

### Overview

The RAG (Retrieval-Augmented Generation) pipeline grounds agent outputs in retrievable text chunks, reducing hallucinations and ensuring verifiable responses.

### Architecture

**Location**: `rag_pipeline.py`

**Key Components**:

1. **Embedding Model**: `all-MiniLM-L6-v2` (384-dim vectors)
2. **Chunking**: Overlapping chunks of ~300 words with 50% overlap
3. **Index**: FAISS (with numpy fallback if FAISS unavailable)
4. **Persistence**: Serialized with pickle to `./data/faiss_index.pkl`

### Usage

```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()

# Build index from papers
papers = [...]  # List of paper dicts
rag.build_index(papers)

# Query the index
top_chunks = rag.query("How do transformers work?", top_k=5)

# Get formatted context for LLM
context = rag.get_context_string("transformer architecture")
```

### Chunking Strategy

- **Chunk size**: ~300 words
- **Overlap**: 50% (150 words)
- **Max chunks per paper**: 3
- **Fallback**: If abstract empty, use title

### Similarity Search

The pipeline supports two backends:

1. **FAISS** (preferred): Fast L2 distance search
2. **NumPy** (fallback): Cosine similarity via normalized embeddings

The backend is selected automatically based on FAISS availability.

### Persistence

Indices are saved and can be loaded:

```python
rag.load_index()  # Load previously saved index
```

---

##  Evaluation & Benchmarking

### Evaluation Module (`evaluator.py`)

Compares multi-agent AMARA output vs. single-LLM baseline using:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **BLEU**: Precision-based similarity
- **Response length**: Word count metrics

### Running Evaluations

```bash
python evaluator.py
```

### Programmatic Usage

```python
from evaluator import EvaluationRunner

runner = EvaluationRunner()

result = runner.evaluate_single_query(
    query="What is federated learning?",
    ground_truth="Federated learning is a decentralized..."  # optional
)

print(f"AMARA ROUGE-1: {result['amara_metrics']['rouge1_f']}")
print(f"Baseline ROUGE-1: {result['baseline_metrics']['rouge1_f']}")
print(f"BLEU: {result['amara_metrics']['bleu']}")
```

### Benchmark Results

The system saves results to `benchmark.json` with:
- Per-query metrics (ROUGE, BLEU)
- Comparison against baseline
- Execution time
- Number of papers retrieved

Example output:
```json
{
    "queries": [
        {
            "query": "...",
            "amara_answer": "...",
            "baseline_answer": "...",
            "amara_metrics": {
                "rouge1_f": 0.65,
                "rouge2_f": 0.42,
                "rougeL_f": 0.58,
                "bleu": 0.51,
                "response_length_words": 750
            },
            "baseline_metrics": {...}
        }
    ],
    "summary": {
        "avg_amara_rouge1": 0.68,
        "avg_baseline_rouge1": 0.52,
        "improvement_percent": 30.8
    }
}
```

---

## Project Structure

```
agentic_project/
├── app.py                    # Streamlit web interface
├── config.py                 # Configuration & LLM setup
├── pipeline.py               # LangGraph orchestration
├── 
├── agents/
│   ├── searcher_agent.py     # Paper retrieval (ArXiv, Semantic Scholar)
│   ├── summarizer_agent.py   # Structured summarization
│   ├── critic_agent.py       # Critical analysis & gaps
│   └── writer_agent.py       # Final synthesis & citations
│
├── rag/
│   └── rag_pipeline.py       # FAISS vector index & retrieval
│
├── evaluator.py              # Evaluation framework (ROUGE, BLEU)
├── benchmark.json            # Evaluation results
│
├── requirements.txt          # Python dependencies
├── env.example               # Environment template
├── README.md                 # This file
└── data/
    └── faiss_index.pkl       # Persisted vector index (auto-generated)
```

### Key Files

| File | Purpose |
|------|---------|
| **app.py** | Streamlit UI with session state management and progress tracking |
| **pipeline.py** | LangGraph StateGraph defining agent orchestration and state schema |
| **config.py** | Environment loading and LLM client instantiation |
| **searcher_agent.py** | Paper retrieval via ArXiv and Semantic Scholar APIs |
| **summarizer_agent.py** | LLM-powered academic summarization |
| **critic_agent.py** | Literature gap analysis and critical review |
| **writer_agent.py** | Final report synthesis with APA citations |
| **rag_pipeline.py** | FAISS-based vector retrieval for grounding |
| **evaluator.py** | Benchmarking and metrics computation |

---

##  Development

### Code Organization

**Import Structure**:
```python
# Config & LLM
from config import get_llm, LLM_PROVIDER

# Agents
from agents.searcher_agent import SearcherAgent
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent
from agents.writer_agent import WriterAgent

# RAG
from rag.rag_pipeline import RAGPipeline

# Pipeline
from pipeline import build_graph, AMARState
```

### Key Design Patterns

1. **State-Driven**: All agents operate on a shared `TypedDict` state
2. **Error Resilience**: Try-except blocks with informative error messages
3. **Modular Agents**: Each agent is independent with a `run(state) → state` interface
4. **RAG Grounding**: All outputs use retrieved chunks to prevent hallucination
5. **Configuration Centralization**: All settings in `config.py` and `.env`

### Extending the System

#### Adding a New Agent

1. Create `agents/new_agent.py`:
```python
from config import get_llm
from langchain.schema import HumanMessage, SystemMessage

class NewAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.3)
    
    def run(self, state: Dict) -> Dict:
        # Process state
        # Call LLM
        # Return updated state
        return {...state, "new_field": result}
```

2. Add to `pipeline.py`:
```python
from agents.new_agent import NewAgent

def new_agent_node(state):
    agent = NewAgent()
    return agent.run(state)

graph.add_node("new_agent", new_agent_node)
graph.add_edge("previous_agent", "new_agent")
```

#### Customizing Search

Edit `searcher_agent.py` to add new sources:
```python
def search_custom_api(self, keywords):
    # Implement API calls
    # Return papers list
    pass
```

#### Modifying RAG Behavior

In `rag_pipeline.py`:
- Change `EMBED_MODEL` to use different embeddings
- Adjust `chunk_size` and overlap percentages
- Switch `_build_faiss_index` implementation

---

##  Contributing

### Reporting Issues

Please open an issue on GitHub with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, LLM provider)

### Submitting Improvements

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes with clear commit messages
4. Ensure code follows existing style
5. Test thoroughly
6. Submit a pull request with description

### Code Style

- **Python**: PEP 8, type hints where possible
- **Docstrings**: Module and class docstrings with purpose
- **Comments**: Inline comments for non-obvious logic
- **Naming**: Clear, descriptive names (avoid abbreviations)

### Testing

```bash
# Test individual agents
python -c "from agents.searcher_agent import SearcherAgent; agent = SearcherAgent(); print(agent.run('test query'))"

# Test pipeline
python -c "from pipeline import build_graph; g = build_graph().compile(); print(g.invoke({'query': 'test'}))"

# Run evaluations
python evaluator.py
```

---

##  License

This project is licensed under the MIT License – see LICENSE file for details.

---

##  Acknowledgments

- **LangChain & LangGraph**: Agent and orchestration framework
- **Groq, OpenAI & Anthropic**: Large language models
- **ArXiv & Semantic Scholar**: Academic paper repositories
- **FAISS**: Efficient similarity search
- **Streamlit**: Web UI framework
- **sentence-transformers**: Embedding models

---

##  Support

For questions or issues:
- **Documentation**: See this README
- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions

---

**Last Updated**: May 2026
**Maintained by**: Syed Ahmed Ishaq
