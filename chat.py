from src.rag import RAG

rag = RAG()
print("ROS 2 assistant ready. Ctrl+C to quit.\n")
while True:
    q = input("You: ").strip()
    if not q:
        continue
    answer, hits = rag.ask(q)
    print(f"\n{answer}\n\nSources:")
    for n, h in enumerate(hits, 1):
        print(f"  [{n}] {h['title']} > {h['section']}  ({h['score']:.2f})  {h['url']}")
    print()
