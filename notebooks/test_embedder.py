from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document

# Инициализация эмбеддера
embedder = OpenAIEmbeddings(
    model="intfloat34/multilingual-e5-large-instruct", 
    base_url="http://localhost:8080/v1",
    api_key="EMPTY"
)

print(embedder.embed_query("Hello, motherfucker!"))

