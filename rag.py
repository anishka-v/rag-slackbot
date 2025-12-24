import os
from io import BytesIO
from typing import Optional, List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF parsing
from pypdf import PdfReader


# --------- Global RAG objects (in-memory) ---------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment variables.")

model = ChatOpenAI(model="gpt-4.1", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


# --------- Helpers ---------

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    parts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            parts.append(f"\n\n[PAGE {i+1}]\n{t}")
    return "\n".join(parts).strip()


def _bytes_to_text(file_bytes: bytes, mimetype: str) -> str:
    mt = (mimetype or "").lower()

    if "pdf" in mt:
        return _extract_text_from_pdf(file_bytes)

    # default: treat as text-ish
    return file_bytes.decode("utf-8", errors="ignore").strip()


# --------- Public API used by Slack bot ---------

def index_slack_file_bytes(
    file_bytes: bytes,
    file_obj: dict,
    slack_channel: Optional[str] = None,
) -> List[str]:
    """
    Convert file bytes -> text -> split -> embed -> add to vector_store.
    Returns list of document IDs inserted into the vector store.
    """
    mimetype = (file_obj.get("mimetype") or "").lower()
    file_id = file_obj.get("id")
    name = file_obj.get("name") or file_obj.get("title") or (file_id or "unknown")

    text = _bytes_to_text(file_bytes, mimetype)
    if not text:
        return []

    base_doc = Document(
        page_content=text,
        metadata={
            "source": "slack",
            "slack_file_id": file_id,
            "slack_filename": name,
            "slack_channel": slack_channel,
            "mimetype": mimetype,
        },
    )

    splits = text_splitter.split_documents([base_doc])
    return vector_store.add_documents(splits)


def answer_query(query: str, slack_channel: Optional[str] = None, k: int = 4) -> str:
    """
    Similarity search -> prompt model with retrieved chunks -> return answer.
    """
    retrieved = vector_store.similarity_search(query, k=k)

    if not retrieved:
        return "I don’t have any indexed documents yet. Upload a file first."

    # Build compact context
    context_parts = []
    for d in retrieved:
        fname = d.metadata.get("slack_filename", "unknown")
        fid = d.metadata.get("slack_file_id", "")
        context_parts.append(f"FILE: {fname} ({fid})\n{d.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    resp = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You answer questions using ONLY the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context}",
            },
        ]
    )
    return resp.content
