from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


@dataclass(slots=True)
class RuntimeDeps:
    Chroma: Any
    HuggingFaceEmbeddings: Any
    PyPDFLoader: Any
    RecursiveCharacterTextSplitter: Any
    ChatPromptTemplate: Any
    Tool: Any
    initialize_agent: Any
    AgentType: Any
    ChatOllama: Any
    LlamaCpp: Any


def _load_dependencies() -> RuntimeDeps | None:
    try:
        from langchain.agents import AgentType, Tool, initialize_agent
        from langchain.prompts import ChatPromptTemplate
        from langchain_chroma import Chroma
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.llms import LlamaCpp
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:
        st.error(
            "Missing Streamlit/LangChain dependencies.\n\n"
            "Install with: `pip install -r requirements-streamlit.txt`\n"
            "(optional) add `llama-cpp-python` for llama.cpp backend.\n\n"
            f"Import error: {exc}"
        )
        return None

    return RuntimeDeps(
        Chroma=Chroma,
        HuggingFaceEmbeddings=HuggingFaceEmbeddings,
        PyPDFLoader=PyPDFLoader,
        RecursiveCharacterTextSplitter=RecursiveCharacterTextSplitter,
        ChatPromptTemplate=ChatPromptTemplate,
        Tool=Tool,
        initialize_agent=initialize_agent,
        AgentType=AgentType,
        ChatOllama=ChatOllama,
        LlamaCpp=LlamaCpp,
    )


def scan_pdf_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(folder.rglob("*.pdf"), key=lambda p: str(p).lower())


def build_vector_db(
    deps: RuntimeDeps,
    pdf_files: list[Path],
    embedding_model: str,
    persist_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
) -> tuple[int, int]:
    pages = []
    for pdf in pdf_files:
        loader = deps.PyPDFLoader(str(pdf))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = str(pdf)
        pages.extend(loaded)

    splitter = deps.RecursiveCharacterTextSplitter(
        chunk_size=max(200, chunk_size),
        chunk_overlap=max(0, min(chunk_overlap, chunk_size // 2)),
    )
    chunks = splitter.split_documents(pages)

    embeddings = deps.HuggingFaceEmbeddings(model_name=embedding_model)
    vector_db = deps.Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    vector_db.delete_collection()
    vector_db = deps.Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    vector_db.add_documents(chunks)
    return len(pages), len(chunks)


def load_vector_db(deps: RuntimeDeps, embedding_model: str, persist_dir: Path, collection_name: str):
    embeddings = deps.HuggingFaceEmbeddings(model_name=embedding_model)
    return deps.Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def create_llm(deps: RuntimeDeps, backend: str, model: str, ollama_url: str, temperature: float, max_tokens: int):
    if backend == "ollama":
        return deps.ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=float(temperature),
            num_predict=int(max_tokens),
        )
    return deps.LlamaCpp(
        model_path=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        n_ctx=8192,
        verbose=False,
    )


def rag_answer(deps: RuntimeDeps, llm, retriever, question: str, top_k: int) -> tuple[str, list]:
    docs = retriever.get_relevant_documents(question)[:top_k]
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = deps.ChatPromptTemplate.from_template(
        "You are a grounded PDF assistant. Use only the context. "
        "If missing, say 'Not found in loaded PDFs.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    rendered = prompt.format(context=context, question=question)

    result = llm.invoke(rendered)
    text = result.content if hasattr(result, "content") else str(result)
    return text, docs


def agent_answer(deps: RuntimeDeps, llm, retriever, question: str, top_k: int) -> tuple[str, list]:
    recent_hits = {"docs": []}

    def search_docs(query: str) -> str:
        docs = retriever.get_relevant_documents(query)[:top_k]
        recent_hits["docs"] = docs
        if not docs:
            return "No matching chunks found."
        lines = []
        for i, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            snippet = " ".join(doc.page_content.split())[:500]
            lines.append(f"[{i}] {src} (page {page})\n{snippet}")
        return "\n\n".join(lines)

    tools = [
        deps.Tool(
            name="pdf_search",
            func=search_docs,
            description="Search loaded PDF chunks and return grounded snippets with sources.",
        )
    ]

    try:
        agent = deps.initialize_agent(
            tools=tools,
            llm=llm,
            agent=deps.AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )
        answer = agent.run(question)
        return str(answer), recent_hits["docs"]
    except Exception:
        return rag_answer(deps, llm, retriever, question, top_k)


def render_sources(docs: list) -> None:
    if not docs:
        st.caption("No citations.")
        return
    for doc in docs:
        src = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        st.caption(f"- {src} (page {page})")


def ensure_state_defaults() -> None:
    st.session_state.setdefault("vector_ready", False)
    st.session_state.setdefault("chat_history", [])


def main() -> None:
    st.set_page_config(page_title="Workflow Streamlit Option", layout="wide")
    st.title("Streamlit Option: LangChain + HuggingFace + Chroma + PyPDFLoader")
    st.caption("Optional prototype alongside FastAPI web app. Local-only model runtimes: Ollama or llama.cpp.")

    ensure_state_defaults()
    deps = _load_dependencies()
    if deps is None:
        return

    with st.sidebar:
        st.subheader("Data + Index")
        data_folder = Path(st.text_input("PDF folder", r"C:\Users\admin\Downloads\Fine Tunining Datasets\train").strip())
        persist_dir = Path(st.text_input("Chroma persist dir", "data/chroma_streamlit").strip())
        collection_name = st.text_input("Collection name", "workflow_streamlit")
        embedding_model = st.text_input("HuggingFace embedding model", "sentence-transformers/all-MiniLM-L6-v2")
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1200, step=100)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1200, value=150, step=10)

        st.subheader("LLM Runtime")
        backend = st.selectbox("Backend", ["ollama", "llamacpp"], index=0)
        ollama_url = st.text_input("Ollama base URL", "http://127.0.0.1:11434")
        default_model = "deepseek-r1:8b" if backend == "ollama" else r"D:\models\model.gguf"
        model = st.text_input("Model name/path", default_model)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        max_tokens = st.number_input("Max tokens", min_value=64, max_value=4096, value=512, step=32)

        st.subheader("Query")
        top_k = st.slider("Top K retrieval", min_value=2, max_value=15, value=6)
        mode = st.selectbox("Mode", ["RAG chain", "Agentic (experimental)"])

        if st.button("Build / Rebuild Chroma", use_container_width=True):
            pdf_files = scan_pdf_files(data_folder)
            if not pdf_files:
                st.error(f"No PDFs found in {data_folder}")
            else:
                with st.spinner("Building vector store..."):
                    pages, chunks = build_vector_db(
                        deps=deps,
                        pdf_files=pdf_files,
                        embedding_model=embedding_model,
                        persist_dir=persist_dir,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        collection_name=collection_name,
                    )
                st.session_state["vector_ready"] = True
                st.success(f"Indexed {len(pdf_files)} PDFs, {pages} pages, {chunks} chunks.")

        if st.button("Load Existing Chroma", use_container_width=True):
            try:
                db = load_vector_db(deps, embedding_model, persist_dir, collection_name)
                db.get()
                st.session_state["vector_ready"] = True
                st.success("Loaded existing Chroma collection.")
            except Exception as exc:
                st.error(f"Could not load Chroma collection: {exc}")

    st.divider()

    if not st.session_state["vector_ready"]:
        st.info("Build or load Chroma index first.")
        return

    question = st.text_area("Ask a question", placeholder="Ask from loaded PDFs...", height=110)
    ask = st.button("Ask", type="primary")

    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Generating answer..."):
                db = load_vector_db(deps, embedding_model, persist_dir, collection_name)
                retriever = db.as_retriever(search_kwargs={"k": int(top_k)})
                llm = create_llm(deps, backend, model, ollama_url, temperature, int(max_tokens))
                if mode == "Agentic (experimental)":
                    answer, docs = agent_answer(deps, llm, retriever, question, int(top_k))
                else:
                    answer, docs = rag_answer(deps, llm, retriever, question, int(top_k))

            st.session_state["chat_history"].append({"q": question, "a": answer, "docs": docs})

    for turn in reversed(st.session_state["chat_history"]):
        with st.container(border=True):
            st.markdown(f"**You:** {turn['q']}")
            st.markdown(f"**Assistant:** {turn['a']}")
            with st.expander("Sources"):
                render_sources(turn["docs"])


if __name__ == "__main__":
    main()
