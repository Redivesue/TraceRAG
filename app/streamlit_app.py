"""Streamlit demo for Perplexity Lite."""

from pathlib import Path
import sys

import streamlit as st

# Streamlit resolves imports relative to the script directory, so add the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.runtime import build_pipeline, get_runtime_metadata
from src.core.schemas import SearchRequest


st.set_page_config(page_title="Perplexity Lite", layout="wide")
st.title("Perplexity Lite")
st.caption("Retrieval-augmented QA with reranking and citation grounding")


@st.cache_resource(show_spinner="Loading retrieval pipeline and indexes...")
def get_pipeline():
    return build_pipeline()


def render_path_notice(trace: dict[str, object], generator_mode: str) -> None:
    reason = str(trace.get("path_reason", ""))
    selected_path = str(trace.get("selected_path", ""))
    if selected_path == "fast_path":
        st.success("Fast path matched. The answer was generated through the low-latency structured path.")
        return
    if reason == "hosted_llm_success":
        st.info("Slow path used hosted LLM generation with grounded evidence.")
        return
    if reason == "llm_not_configured_offline_fallback":
        st.warning("Hosted LLM is not configured. The system used offline grounded fallback.")
        return
    if reason == "hosted_llm_error_offline_fallback":
        st.warning("Hosted LLM request failed or timed out. The system fell back to offline grounded generation.")
        return
    if reason == "no_evidence":
        st.warning("No supporting evidence was retrieved, so the system returned a conservative answer.")
        return
    if generator_mode == "offline":
        st.info("Slow path used offline grounded fallback.")


if "query" not in st.session_state:
    st.session_state.query = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_show_debug" not in st.session_state:
    st.session_state.last_show_debug = True
if "last_top_k" not in st.session_state:
    st.session_state.last_top_k = 5

st.caption("The page renders first and only initializes the retrieval stack when you click Search.")

with st.expander("Runtime Status", expanded=False):
    runtime = get_runtime_metadata()
    manifest = runtime.get("index_manifest", {})
    st.markdown(
        "\n".join(
            [
                f"- LLM provider: `{runtime.get('llm_provider', '')}`",
                f"- LLM model: `{runtime.get('llm_model', '')}`",
                f"- Dense backend: `{runtime.get('dense_encoder_backend', '')}`",
                f"- Dense model: `{runtime.get('dense_model_name', '')}`",
                f"- Chunk path: `{runtime.get('default_chunk_path', '')}`",
                f"- Index manifest status: `{manifest.get('status', 'available')}`",
                f"- Chunk strategy: `{manifest.get('chunk_strategy', '')}`",
                f"- Encoder backend: `{manifest.get('encoder_backend', '')}`",
                f"- Encoder model: `{manifest.get('encoder_model', '')}`",
            ]
        )
    )
    if manifest:
        st.json(manifest)

with st.expander("Ops Note", expanded=False):
    st.markdown(
        "\n".join(
            [
                "- `fast_path` is a narrow latency optimization layer for simple structured entity questions.",
                "- `slow_path` is the default grounded QA path: retrieval, fusion, dedup, rerank, then generation.",
                "- If hosted LLM generation is unavailable, the system falls back to offline grounded generation.",
                "- If retrieved evidence is insufficient, the system should prefer conservative answers over speculation.",
                "- Runtime and index metadata are shown in `Runtime Status`; per-request path decisions are shown in `Request Trace`.",
            ]
        )
    )

examples = [
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What did the Pound–Rebka experiment test?",
    "What is the purpose of the Nucifer experiment?",
]
example_query = st.selectbox("Example queries", options=[""] + examples, index=0)
if example_query and example_query != st.session_state.query:
    st.session_state.query = example_query

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Ask a question",
        key="query",
        placeholder="Type a question and click Search",
    )
    top_k = st.slider("Top-k evidence", min_value=3, max_value=10, value=st.session_state.last_top_k)
    show_debug = st.checkbox("Show retrieved evidence", value=st.session_state.last_show_debug)
    submitted = st.form_submit_button("Search")

if submitted and query.strip():
    pipeline = get_pipeline()
    st.session_state.last_top_k = top_k
    st.session_state.last_show_debug = show_debug
    st.session_state.last_response = pipeline.run(
        SearchRequest(query=query.strip(), top_k=top_k, debug=show_debug)
    )
    st.caption(f"Generator mode: `{pipeline.generator.provider_name}`")

response = st.session_state.last_response
if response is not None:
    show_debug = st.session_state.last_show_debug
    trace = response.trace or {}
    render_path_notice(trace, response.generator_mode)
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Answer")
        st.write(response.answer)
        st.caption(f"Generated by: `{response.generator_mode}`")
        st.caption(
            " | ".join(
                [
                    f"retrieve={response.timings.get('retrieval_seconds', 0):.2f}s",
                    f"rerank={response.timings.get('rerank_seconds', 0):.2f}s",
                    f"generate={response.timings.get('generation_seconds', 0):.2f}s",
                    f"total={response.timings.get('total_seconds', 0):.2f}s",
                ]
            )
        )
        st.caption(
            " | ".join(
                [
                    f"request_id={response.request_id}",
                    f"path={trace.get('selected_path', '')}",
                    f"reason={trace.get('path_reason', '')}",
                ]
            )
        )
    with right:
        st.subheader("Citations")
        if response.citations:
            for citation in response.citations:
                st.markdown(f"[{citation.label}] {citation.title}")
        else:
            st.caption("No citations returned.")

    with st.expander("Request Trace", expanded=False):
        trace = response.trace or {}
        st.markdown(
            "\n".join(
                [
                    f"- Request ID: `{response.request_id}`",
                    f"- Selected path: `{trace.get('selected_path', '')}`",
                    f"- Path reason: `{trace.get('path_reason', '')}`",
                    f"- Generator mode: `{trace.get('generator_mode', response.generator_mode)}`",
                    f"- Retrieved candidates: `{trace.get('retrieved_candidate_count', 0)}`",
                    f"- Reranked candidates: `{trace.get('reranked_candidate_count', 0)}`",
                    f"- Cache hit: `{trace.get('cache_hit', 0)}`",
                ]
            )
        )
        st.json(trace)

    if show_debug:
        st.subheader("Reranked Evidence")
        for item in response.reranked:
            st.markdown(f"**{item.rank}. {item.chunk.title}**")
            st.caption(
                f"rerank={item.rerank_score:.4f} | retrieval={item.retrieval_score:.4f}"
            )
            st.write(item.chunk.text)

        st.subheader("Retrieved Candidates")
        for item in response.retrieved:
            st.markdown(f"**{item.chunk.title}**")
            st.caption(f"source={item.source_method} | score={item.score:.4f}")
elif not submitted:
    st.info("Enter a question or choose an example, then click Search.")
