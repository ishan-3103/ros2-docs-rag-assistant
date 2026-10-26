"""Build the vector index:  python -m src.ingest"""
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config as C
from src.chunking import chunk_rst


def load_chunks():
    all_chunks = []
    for folder in C.INCLUDE_DIRS:
        for path in sorted((C.DOCS_DIR / folder).rglob("*.rst")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = path.relative_to(C.DOCS_DIR)
            url = f"{C.DOCS_BASE_URL}/{rel.with_suffix('.html').as_posix()}"
            chunks = chunk_rst(text, C.MAX_CHARS, C.OVERLAP_CHARS)
            title = chunks[0]["section"] if chunks and chunks[0]["section"] else path.stem
            for ch in chunks:
                if len(ch["text"]) < 80:      # drop tiny fragments
                    continue
                ch.update(source=rel.as_posix(), url=url, title=title)
                all_chunks.append(ch)
    return all_chunks


def main():
    chunks = load_chunks()
    print(f"{len(chunks)} chunks")
    model = SentenceTransformer(C.EMBED_MODEL)
    texts = [f"{c['title']} > {c['section']}\n{c['text']}" for c in chunks]
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])   # inner product == cosine (vectors normalized)
    index.add(emb)
    faiss.write_index(index, str(C.INDEX_PATH))
    C.CHUNKS_PATH.write_text(json.dumps(chunks), encoding="utf-8")
    print("Saved index and chunks.")


if __name__ == "__main__":
    main()
