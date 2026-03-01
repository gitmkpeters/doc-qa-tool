from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os

# Configure to use local Ollama models
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="llama3.2")

# Set up ChromaDB
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build or load index
if chroma_collection.count() == 0:
    print("First run — loading and indexing document...")
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("Index saved to disk!")
else:
    print("Loading existing index from disk...")
    index = VectorStoreIndex.from_vector_store(vector_store)

print("Ready! Ask your questions (type 'quit' to exit)\n")

query_engine = index.as_query_engine()

while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break
    response = query_engine.query(question)
    print(f"\nAnswer: {response}\n")