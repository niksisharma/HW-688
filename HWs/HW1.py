import io
import time
import streamlit as st
from typing import Optional
from openai import OpenAI

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

MODELS = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"]


def read_all(uploaded_file) -> bytes:
    data = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    return data


def read_pdf(uploaded_file) -> str:
    data = read_all(uploaded_file)

    if HAS_FITZ:
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text.strip()

    raise RuntimeError("Install PyMuPDF.")


def load_document(uploaded_file) -> Optional[str]:
    # Accepting only .txt or .pdf
    if uploaded_file is None:
        return None

    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "txt":
        text = uploaded_file.read().decode(errors="ignore")
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return text
    elif file_extension == "pdf":
        return read_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return None


def build_messages(doc: str, question: str):
    content = (
        "Here is a document:\n\n"
        f"{doc}\n\n"
        "---\n\n"
        "Answer the following question concisely using only information from the document.\n"
        f"Question: {question}"
    )
    return [{"role": "user", "content": content}]


def ask_once(client: OpenAI, model: str, messages):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=model, messages=messages)
    latency = round(time.perf_counter() - t0, 2)
    answer = resp.choices[0].message.content if resp.choices else "(no answer)"
    usage = getattr(resp, "usage", None)
    prompt_toks = getattr(usage, "prompt_tokens", None) if usage else None
    completion_toks = getattr(usage, "completion_tokens", None) if usage else None
    total_toks = getattr(usage, "total_tokens", None) if usage else None
    return answer, latency, prompt_toks, completion_toks, total_toks


def main():
    st.set_page_config(page_title="HW1", page_icon="📄")
    st.title("HW1 Doc Q&A - 4 Model Comparison: Nikita Sharma ")

    if "doc_text" not in st.session_state:
        st.session_state["doc_text"] = None
        st.session_state["doc_name"] = None
        st.session_state["results"] = []

    # API key input
    api_key = st.text_input("OpenAI API Key", type="password", help="Paste your OpenAI key here")
    client = OpenAI(api_key=api_key) if api_key else None

    # File uploading 
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))

    # If file is removed, app will forget the data
    if uploaded_file is None and st.session_state["doc_text"] is not None:
        st.session_state["doc_text"] = None
        st.session_state["doc_name"] = None

    # If new file, reloads the text
    if uploaded_file is not None and uploaded_file.name != st.session_state["doc_name"]:
        text = load_document(uploaded_file)
        if text:
            st.session_state["doc_text"] = text
            st.session_state["doc_name"] = uploaded_file.name
        else:
            st.session_state["doc_text"] = None
            st.session_state["doc_name"] = None

    if st.session_state["doc_text"]:
        st.success(f"Loaded: {st.session_state['doc_name']}")
        with st.expander("Preview (first 800 chars)"):
            st.text(st.session_state["doc_text"][:800])

    question = st.text_input("Question", value="Is this course hard?")

    # Run button
    if st.button("Run on all 4 models"):
        st.session_state["results"] = []
        if not client:
            st.error("Please enter your OpenAI API key.")
        elif not st.session_state["doc_text"]:
            st.error("Please upload a .txt or .pdf document.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            msgs = build_messages(st.session_state["doc_text"], question)
            for m in MODELS:
                with st.spinner(f"Running {m}..."):
                    try:
                        ans, lat, pt, ct, tt = ask_once(client, m, msgs)
                    except Exception as e:
                        ans, lat, pt, ct, tt = (f"Error: {e}", None, None, None, None)
                st.session_state["results"].append({
                    "model": m,
                    "answer": ans,
                    "latency_s": lat,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt
                })

    # Showing results
    for r in st.session_state["results"]:
        st.markdown(f"### {r['model']}")
        st.write(r["answer"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency (s)", r["latency_s"])
        c2.metric("Input tokens", r["prompt_tokens"])
        c3.metric("Output tokens", r["completion_tokens"])
        c4.metric("Total tokens", r["total_tokens"])


if __name__ == "__main__":
    main()
