from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


embedder = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B", 
    base_url="http://localhost:8090/v1",
    api_key="EMPTY"
)

