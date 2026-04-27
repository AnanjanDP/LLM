import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#  Files for persistence
INDEX_FILE = "faiss.index"
DOC_FILE = "documents.npy"

#  In-memory storage
index = None
documents = []


#  Load existing index + documents
if os.path.exists(INDEX_FILE) and os.path.exists(DOC_FILE):
    index = faiss.read_index(INDEX_FILE)
    documents = list(np.load(DOC_FILE, allow_pickle=True))


#  Chunking with overlap (better retrieval)
def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


#  Add document to vector DB
def add_document(text: str):
    global index

    chunks = chunk_text(text)

    if len(chunks) == 0:
        return

    embeddings = model.encode(chunks)

    #  Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    if index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

    index.add(np.array(embeddings).astype("float32"))
    documents.extend(chunks)

    #  Save to disk
    faiss.write_index(index, INDEX_FILE)
    np.save(DOC_FILE, documents)


#  Retrieve relevant chunks
def get_context(query: str, k=3):
    if index is None or len(documents) == 0:
        return [], "No relevant context available."

    query_embedding = model.encode([query])

    #  Normalize query
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), k
    )

    results = []
    for i in indices[0]:
        if 0 <= i < len(documents):
            results.append(documents[i])

    #  Return BOTH sources + combined context
    return results, "\n".join(results)