from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document

# Инициализация эмбеддера
embedder = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B", 
    base_url="http://localhost:8090/v1",
    api_key="EMPTY"
)

print(embedder.embed_query("Hello, motherfucker!"))

