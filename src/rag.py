import json
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

from src import config as C

SYSTEM = (
    "If user greets you, greet them back and ask how can you help them, and dont cite sources at that time."
    "You are a ROS 2 documentation assistant. Answer ONLY using the numbered context. "
    "Cite sources like [1], [2]. Keep code snippets intact. "
    "If the context does not contain the answer, say you don't know." 
)


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(C.EMBED_MODEL)
        self.index = faiss.read_index(str(C.INDEX_PATH))
        self.chunks = json.loads(C.CHUNKS_PATH.read_text(encoding="utf-8"))

    def search(self, query: str, k: int = C.TOP_K):
        q = self.model.encode([C.QUERY_PREFIX + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [{**self.chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


def build_prompt(question, hits):
    ctx = "\n\n".join(
        f"[{n}] ({h['title']} > {h['section']})\n{h['text']}" for n, h in enumerate(hits, 1)
    )
    return f"Context:\n{ctx}\n\nQuestion: {question}"


class RAG:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = Groq()  # reads GROQ_API_KEY from env

    def ask(self, question: str, k: int = C.TOP_K):
        hits = self.retriever.search(question, k)
        resp = self.llm.chat.completions.create(
            model=C.LLM_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(question, hits)},
            ],
        )
        return resp.choices[0].message.content, hits
