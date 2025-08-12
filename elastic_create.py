import asyncio
import json
import os
from typing import List, Dict, Optional
import boto3
import requests
from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch

# Инициализация клиента Elasticsearch
es = AsyncElasticsearch(
    "http://localhost:9200", 
    basic_auth=("elastic", "JxR_8Fj6P23J_q_xU-rf"),
    headers={"Accept": "application/vnd.elasticsearch+json;compatible-with=8"}
)

class TextEmbeddingsInferenceEmbedder:
    """Прямой клиент для text-embeddings-inference API"""
    
    def __init__(self, base_url: str = "http://localhost:8080", model: str = "intfloat/multilingual-e5-large-instruct"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создать эмбеддинги для списка текстов"""
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": texts},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return [item["embedding"] for item in result["data"]]
    
    def embed_query(self, text: str) -> List[float]:
        """Создать эмбеддинг для одного текста"""
        return self.embed_documents([text])[0]

# Инициализация эмбеддера
embedder = TextEmbeddingsInferenceEmbedder()

def _get_s3_client() -> boto3.client:
    load_dotenv()
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    # Поддержка двух схем имен переменных окружения
    access_key = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("S3_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

    client_kwargs: Dict[str, Optional[str]] = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    if region:
        client_kwargs["region_name"] = region
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
    if session_token:
        client_kwargs["aws_session_token"] = session_token

    return boto3.client("s3", **client_kwargs)

def _list_s3_json_keys(s3_client: boto3.client, bucket: str, prefix: Optional[str] = None) -> List[str]:
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and key.endswith(".json"):
                keys.append(key)
    return keys

def _read_json_from_s3(s3_client: boto3.client, bucket: str, key: str) -> Dict:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    return json.loads(body.decode("utf-8"))

async def create_index_mapping(index_name: str = "scientific_papers"):
    """Создает индекс с маппингом для векторного поиска"""
    
    mapping = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "integer"},
                "text": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "text_embedding": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "cosine"
                },
                "chunk_index": {"type": "integer"},
                "total_chunks": {"type": "integer"},
                "length_tokens": {"type": "integer"},
                "type": {"type": "keyword"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    try:
        try:
            await es.indices.delete(index=index_name)
        except Exception:
            pass
        
        await es.indices.create(index=index_name, **mapping)
        print(f"✅ Индекс '{index_name}' успешно создан")
        
    except Exception as e:
        print(f"❌ Ошибка при создании индекса: {e}")

async def process_and_index_chunks(chunks: List[Dict], index_name: str = "scientific_papers", batch_size: int = 10):
    """Обрабатывает чанки, создает эмбеддинги и индексирует в Elasticsearch"""
    
    print(f"🔄 Начинаем обработку {len(chunks)} чанков...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        texts = [chunk["text"] for chunk in batch]
        
        try:
            print(f"📝 Создаем эмбеддинги для батча {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, embedder.embed_documents, texts
            )
            
            print(f"🔢 Получено эмбеддингов: {len(embeddings)}, размер: {len(embeddings[0]) if embeddings else 0}")
            
            actions = []
            
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                doc = {
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "text_embedding": embedding,
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "length_tokens": chunk["length_tokens"],
                    "type": chunk["type"]
                }
                
                actions.append({
                    "_index": index_name,
                    "_id": chunk["id"],
                    "_source": doc
                })
            
            from elasticsearch.helpers import async_bulk
            await async_bulk(es, actions)
            
            print(f"✅ Проиндексирован батч {i//batch_size + 1}, документы {i}-{min(i + batch_size - 1, len(chunks) - 1)}")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке батча {i//batch_size + 1}: {e}")
            continue

async def load_and_process_data(json_file_path: str):
    """Загружает данные из JSON файла и обрабатывает их"""
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("content", {}).get("chunks", [])
        
        if not chunks:
            print("❌ Чанки не найдены в файле")
            return
        
        print(f"📚 Загружено {len(chunks)} чанков")
        
        await create_index_mapping()
        
        await process_and_index_chunks(chunks)
        
        print("🎉 Индексация завершена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")

async def load_and_process_s3(bucket_name: str, prefix: Optional[str] = None, index_name: str = "scientific_papers"):
    loop = asyncio.get_event_loop()
    s3 = _get_s3_client()

    print(f"📦 Чтение JSON файлов из S3: bucket='{bucket_name}', prefix='{prefix or ''}'")
    keys = await loop.run_in_executor(None, _list_s3_json_keys, s3, bucket_name, prefix)

    if not keys:
        print("❌ В S3 не найдено JSON файлов для обработки")
        return

    print(f"📄 Найдено файлов: {len(keys)}")

    await create_index_mapping(index_name=index_name)

    for idx, key in enumerate(keys, start=1):
        try:
            data = await loop.run_in_executor(None, _read_json_from_s3, s3, bucket_name, key)
            chunks = data.get("content", {}).get("chunks", [])
            if not chunks:
                print(f"⚠️ В файле {key} не найдено чанков, пропуск")
                continue
            print(f"➡️ [{idx}/{len(keys)}] Файл: {key}, чанков: {len(chunks)}")
            await process_and_index_chunks(chunks, index_name=index_name)
        except Exception as e:
            print(f"❌ Ошибка при обработке файла {key}: {e}")
            continue

async def search_similar_text(query: str, index_name: str = "scientific_papers", k: int = 5):
    """Поиск похожих текстов по векторному сходству"""
    
    try:
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, embedder.embed_query, query
        )
        
        search_body = {
            "knn": {
                "field": "text_embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": 100
            },
            "_source": ["text", "chunk_id", "chunk_index"]
        }
        
        response = await es.search(index=index_name, **search_body)
        
        print(f"🔍 Найдено {len(response['hits']['hits'])} похожих документов:")
        
        for i, hit in enumerate(response['hits']['hits']):
            print(f"\n{i+1}. Score: {hit['_score']:.4f}")
            print(f"Chunk ID: {hit['_source']['chunk_id']}")
            print(f"Text: {hit['_source']['text'][:200]}...")
        
        return response['hits']['hits']
        
    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")
        return []

async def main():
    """Основная функция"""
    load_dotenv()
    bucket_name = "palladium-md-to-chunks"
    prefix = os.getenv("S3_PREFIX")
    index_name = os.getenv("ES_INDEX_NAME", "scientific_papers")
    
    try:
        info = await es.info()
        print(f"✅ Подключение к Elasticsearch успешно: {info['version']['number']}")
        
        await load_and_process_s3(bucket_name=bucket_name, prefix=prefix, index_name=index_name)
        
        await search_similar_text("gold effect on creep properties", index_name=index_name)
        
    except Exception as e:
        print(f"❌ Ошибка подключения к Elasticsearch: {e}")
        
    finally:
        await es.close()

async def index_custom_chunks(chunks_data: List[Dict]):
    """Функция для индексации пользовательских данных"""
    
    await create_index_mapping()
    await process_and_index_chunks(chunks_data)

if __name__ == "__main__":
    asyncio.run(main())
