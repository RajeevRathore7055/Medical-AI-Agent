import os
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"


class RAGEngine:
    def __init__(self, pdf_dir: str, txt_dir: str = None):
        self.pdf_dir  = pdf_dir
        self.txt_dir  = txt_dir
        self.chunks   = []
        self.metadata = []
        self.index    = None
        self.embedder = SentenceTransformer(EMBED_MODEL)
        logger.info("✅ RAG Engine initialized")

    # ── Load PDFs ─────────────────────────────────────────────
    def load_pdfs(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        # Load PDF files
        if os.path.exists(self.pdf_dir):
            for fname in os.listdir(self.pdf_dir):
                if fname.endswith(".pdf"):
                    path   = os.path.join(self.pdf_dir, fname)
                    reader = PdfReader(path)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            for i, chunk in enumerate(splitter.split_text(text)):
                                self.chunks.append(chunk)
                                self.metadata.append({
                                    "source_file": fname,
                                    "page_number": page_num + 1,
                                    "chunk_id":    len(self.chunks) - 1
                                })

        # Load TXT files (as fallback)
        if self.txt_dir and os.path.exists(self.txt_dir):
            for fname in os.listdir(self.txt_dir):
                if fname.endswith(".txt"):
                    path = os.path.join(self.txt_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    for i, chunk in enumerate(splitter.split_text(text)):
                        self.chunks.append(chunk)
                        self.metadata.append({
                            "source_file": fname.replace(".txt", ".pdf"),
                            "page_number": 1,
                            "chunk_id":    len(self.chunks) - 1
                        })

        logger.info(f"✅ Loaded {len(self.chunks)} chunks from documents")

    # ── Build FAISS Index ─────────────────────────────────────
    def build_index(self):
        if not self.chunks:
            self.load_pdfs()

        embeddings = self.embedder.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")

        dim         = embeddings.shape[1]
        self.index  = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        logger.info(f"✅ FAISS index built: {self.index.ntotal} vectors")

    # ── Search ────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None:
            self.build_index()

        query_emb = self.embedder.encode(
            [query], convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "content":  self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score":    float(dist)
                })
        return results

    # ── Format Results ────────────────────────────────────────
    def format_context(self, results: list[dict]) -> tuple[str, str]:
        context = ""
        sources = set()
        for r in results:
            context += r["content"] + "\n\n"
            sources.add(r["metadata"]["source_file"])
        return context.strip(), ", ".join(sources)
