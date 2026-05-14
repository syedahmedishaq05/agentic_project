"""
AMARA - Streamlit Web Interface
A polished web UI for interacting with the multi-agent research assistant.
Run with: streamlit run web/app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMARA - Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.4rem;
        font-weight: 600;
        color: #0f172a;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .agent-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .agent-card.active { border-left-color: #10b981; }
    .agent-card.done   { border-left-color: #6366f1; }
    .paper-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-box {
        text-align: center;
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 0.8rem;
    }
    .stButton>button {
        background: #0f172a;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        transition: background 0.2s;
    }
    .stButton>button:hover { background: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    provider = st.selectbox("LLM Provider", ["openai", "anthropic"], index=0)
    os.environ["LLM_PROVIDER"] = provider

    if provider == "openai":
        api_key = st.text_input("OpenAI API Key", type="password",
                                value=os.getenv("OPENAI_API_KEY", ""))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = st.text_input("Anthropic API Key", type="password",
                                value=os.getenv("ANTHROPIC_API_KEY", ""))
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

    max_papers = st.slider("Max Papers to Retrieve", 3, 15, 6)
    os.environ["MAX_PAPERS"] = str(max_papers)

    st.markdown("---")
    st.markdown("### 📋 Pipeline Agents")
    st.markdown("""
    1. 🔍 **Searcher** — ArXiv + Semantic Scholar
    2. 📝 **Summarizer** — Extract key ideas
    3. 🔬 **Critic** — Gap & weakness analysis
    4. ✍️ **Writer** — Synthesize final answer
    """)

    st.markdown("---")
    st.markdown("**AMARA v1.0**  \n*Dept. of AI — 22k-4114, 22k-4102, 22k-4066*")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 AMARA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Autonomous Multi-Agent Research Assistant</div>', unsafe_allow_html=True)
st.markdown("---")

# Example queries
st.markdown("**Example queries:**")
examples = [
    "How does Retrieval-Augmented Generation reduce hallucination in LLMs?",
    "What are the advantages of multi-agent AI systems for complex reasoning?",
    "How does Chain-of-Thought prompting improve LLM performance?",
]
cols = st.columns(3)
for i, ex in enumerate(examples):
    if cols[i].button(f"💡 {ex[:45]}...", key=f"ex_{i}"):
        st.session_state["query_input"] = ex

# Query input
query = st.text_area(
    "Enter your research question:",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="e.g., What are the key challenges in deploying multi-agent LLM systems?",
    key="query_box",
)

run_btn = st.button("🚀 Run AMARA Pipeline", use_container_width=True)

# ── Pipeline Execution ────────────────────────────────────────────────────────
if run_btn and query.strip():
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar.")
        st.stop()

    progress_placeholder = st.empty()
    status_cols          = st.columns(4)
    agent_names          = ["🔍 Searcher", "📝 Summarizer", "🔬 Critic", "✍️ Writer"]
    statuses             = ["⏳ Waiting"] * 4

    def update_status(idx: int, status: str):
        statuses[idx] = status
        with progress_placeholder.container():
            cols = st.columns(4)
            for j, (name, st_) in enumerate(zip(agent_names, statuses)):
                cols[j].markdown(
                    f"**{name}**\n\n{st_}",
                    unsafe_allow_html=True,
                )

    update_status(0, "🟡 Running...")

    # Import pipeline lazily to respect env vars set above
    try:
        from pipeline import AMARAPipeline
        pipeline = AMARAPipeline()

        # We patch the individual agents to update status during run
        from agents.searcher_agent import SearcherAgent
        from agents.summarizer_agent import SummarizerAgent
        from agents.critic_agent import CriticAgent
        from agents.writer_agent import WriterAgent

        # Run with progress updates
        with st.spinner("Running multi-agent pipeline..."):
            searcher = SearcherAgent()
            update_status(0, "🟡 Searching...")
            search_out = searcher.run(query)
            update_status(0, f"✅ {len(search_out['papers'])} papers found")

            update_status(1, "🟡 Summarizing...")
            summarizer = SummarizerAgent()
            sum_out = summarizer.run(search_out)
            update_status(1, "✅ Done")

            update_status(2, "🟡 Analyzing...")
            critic = CriticAgent()
            crit_out = critic.run(sum_out)
            update_status(2, "✅ Done")

            update_status(3, "🟡 Writing...")
            writer = WriterAgent()
            final = writer.run(crit_out)
            update_status(3, "✅ Done")

        st.success("✅ Pipeline complete!")
        st.markdown("---")

        # ── Results Tabs ──────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Final Answer", "📚 Papers", "🔬 Critique", "📊 Stats"])

        with tab1:
            st.markdown("## Research Answer")
            st.markdown(final.get("final_answer", "No answer generated."))

        with tab2:
            st.markdown(f"## Retrieved Papers ({len(final.get('papers', []))})")
            for i, p in enumerate(final.get("papers", []), 1):
                with st.expander(f"{i}. {p['title']} ({p['year']})"):
                    st.markdown(f"**Authors:** {', '.join(p['authors'])}")
                    st.markdown(f"**Source:** {p['source'].replace('_', ' ').title()}")
                    if p.get("url"):
                        st.markdown(f"**URL:** [{p['url']}]({p['url']})")
                    st.markdown("**Summary:**")
                    st.markdown(p.get("summary", p.get("abstract", "N/A")))

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔍 Critical Analysis")
                st.markdown(final.get("critique", ""))
            with col2:
                st.markdown("### 🤝 Consensus & Conflicts")
                st.markdown(final.get("consensus", ""))

        with tab4:
            st.markdown("### Pipeline Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Papers Retrieved",  len(final.get("papers", [])))
            c2.metric("Keywords Used",     len(final.get("keywords", [])))
            c3.metric("Answer Length",     f"{len(final.get('final_answer','').split())} words")
            st.markdown("**Search Keywords:**")
            st.code(", ".join(final.get("keywords", [])))

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)

elif run_btn:
    st.warning("⚠️ Please enter a research question.")
