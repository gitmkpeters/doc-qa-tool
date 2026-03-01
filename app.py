import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os

# Page config
st.set_page_config(
    page_title="JPR AI",
    page_icon="🏦",
    layout="wide"
)

# Configure Ollama
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.text_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=10)

# Sidebar
with st.sidebar:
    st.title("🏦 JPR AI")
    st.caption("Your Financial System Intelligence")
    st.divider()
    
    st.subheader("📁 Add Documents")
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "txt", "php", "sql", "bat", "vbs"]
    )
    
    if uploaded_file:
        save_path = f"docs/{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ {uploaded_file.name} uploaded!")
        st.info("Restart the app to index the new document")
    
    st.divider()
    st.subheader("📚 Indexed Documents")
    docs = os.listdir("docs")
    for doc in docs:
        st.write(f"📄 {doc}")
    
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main area
st.title("🏦 JPR AI Document Assistant")
st.caption("Ask questions about your financial system — powered by local AI")
st.divider()

# Load index
@st.cache_resource
def load_index():
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        embed_batch_size=1
    )
    
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() == 0:
        print("First run — loading and indexing document...")
        documents = SimpleDirectoryReader("docs").load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
    else:
        print("Loading existing index from disk...")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
    
    return index

with st.spinner("Loading JPR AI..."):
    index = load_index()
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize"
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📎 Sources"):
                st.caption(message["sources"])

# Chat input
if prompt := st.chat_input("Ask anything about your system..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            answer = str(response)
            st.markdown(answer)
            
            sources = ""
            if hasattr(response, 'source_nodes'):
                for i, node in enumerate(response.source_nodes):
                    sources += f"**Source {i+1}:** {node.metadata.get('file_name', 'Unknown')} (score: {node.score:.2f})\n\n"
                    sources += f"{node.text[:200]}...\n\n"
            
            if sources:
                with st.expander("📎 View Sources"):
                    st.caption(sources)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            