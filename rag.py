from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load saved FAISS vector database
vectorstore = FAISS.load_local(
    "rag/vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# Function for retrieving relevant information
def search_knowledge(question, k=3):

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results