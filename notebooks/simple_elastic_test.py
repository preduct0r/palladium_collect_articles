import asyncio
from elasticsearch import AsyncElasticsearch
from langchain.embeddings import OpenAIEmbeddings

# Инициализация клиента Elasticsearch
es = AsyncElasticsearch(
    "http://localhost:9200", 
    basic_auth=("elastic", "JxR_8Fj6P23J_q_xU-rf"),
    headers={"Accept": "application/vnd.elasticsearch+json;compatible-with=8"}
)

# Инициализация эмбеддера
embedder = OpenAIEmbeddings(
    model="intfloat/multilingual-e5-large-instruct", 
    base_url="http://localhost:8080/v1",
    api_key="EMPTY"
)

async def quick_search(query: str, k: int = 5):
    """Быстрый поиск без RAG - только семантический поиск"""
    
    try:
        print(f"🔍 Поиск: '{query}'")
        
        # Создаем эмбеддинг
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, embedder.embed_query, query
        )
        
        # Поиск
        search_body = {
            "knn": {
                "field": "text_embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": 100
            },
            "_source": ["text", "chunk_id", "chunk_index"]
        }
        
        response = await es.search(index="scientific_papers", **search_body)
        
        print(f"\n✅ Найдено {len(response['hits']['hits'])} результатов:")
        print("=" * 70)
        
        for i, hit in enumerate(response['hits']['hits'], 1):
            print(f"\n{i}. Score: {hit['_score']:.4f} | Chunk: {hit['_source']['chunk_id']}")
            print(f"Текст: {hit['_source']['text']}")
            print("-" * 70)
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        await es.close()

async def main():
    # Примеры запросов для тестирования
    queries = [
        "gold effect on creep",
        "микроструктура",
        "ползучесть металлов"
    ]
    
    # Выберите запрос (или введите свой)
    query = "Unparalleled Rates for the Activation of Aryl Chlorides and Bromides"  # Измените индекс или введите свой запрос
    
    await quick_search(query, k=3)

if __name__ == "__main__":
    asyncio.run(main()) 