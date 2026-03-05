import chromadb
try:
    client = chromadb.PersistentClient(path="/tmp/test_chroma")
    print("ChromaDB initialized successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
