from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

print("Setting up embedding model...")
embed_model = OllamaEmbedding(model_name="nomic-embed-text", embed_batch_size=1)
splitter = SentenceSplitter(chunk_size=128, chunk_overlap=10)

print("Connecting to ChromaDB...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Loading documents...")
documents = SimpleDirectoryReader("docs").load_data()
nodes = splitter.get_nodes_from_documents(documents)
print(f"Total nodes to index: {len(nodes)}")

print("Indexing... this will take a few minutes")
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

print("Done! Index saved to chroma_db")