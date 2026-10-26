# ROS 2 Docs Assistant (RAG)

Retrieval-augmented chatbot over the ROS 2 documentation, built from scratch
(no LangChain): structure-aware chunking, BGE embeddings, FAISS, Groq LLM.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY

# get the docs (RST source files)
git clone --depth 1 -b humble https://github.com/ros2/ros2_documentation data/ros2_documentation

python -m src.ingest        # build the index (a few minutes first time)
python chat.py              # ask questions
```

## Roadmap
- [x] Phase 1: baseline pipeline
- [ ] Phase 2: BM25 + RRF hybrid search, cross-encoder reranker, query rewriting, streaming UI
- [ ] Phase 3: eval set (30-50 Qs), Hit Rate@k, MRR, config comparison table
- [ ] Phase 4: FastAPI + Docker + HF Spaces deploy
