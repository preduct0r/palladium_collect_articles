import asyncio
import json
import os
from typing import List, Dict, Optional
import boto3
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document

# Инициализация эмбеддера
embedder = OpenAIEmbeddings(
    model="intfloat/multilingual-e5-large-instruct", 
    base_url="http://localhost:8080/v1",
    api_key="EMPTY"
)

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

def load_texts_from_s3(bucket_name: str, prefix: Optional[str] = None) -> List[str]:
    """Загружает тексты из S3 бакета"""
    # s3 = _get_s3_client()
    
    # print(f"📦 Чтение JSON файлов из S3: bucket='{bucket_name}', prefix='{prefix or ''}'")
    # keys = _list_s3_json_keys(s3, bucket_name, prefix)
    
    # if not keys:
    #     print("❌ В S3 не найдено JSON файлов для обработки")
    #     return []
    
    # print(f"📄 Найдено файлов: {len(keys)}")
    
    # texts = []
    # for idx, key in enumerate(keys, start=1):
    #     try:
    #         data = _read_json_from_s3(s3, bucket_name, key)
    #         chunks = data.get("content", {}).get("chunks", [])
    #         if not chunks:
    #             print(f"⚠️ В файле {key} не найдено чанков, пропуск")
    #             continue
            
    #         print(f"➡️ [{idx}/{len(keys)}] Файл: {key}, чанков: {len(chunks)}")
            
    #         # Извлекаем текст из каждого чанка
    #         for chunk in chunks:
    #             if chunk.get("text"):
    #                 texts.append(chunk["text"])
                    
    #     except Exception as e:
    #         print(f"❌ Ошибка при обработке файла {key}: {e}")
    #         continue
    
    # print(f"✅ Всего загружено текстов: {len(texts)}")


    # with open("texts.json", "r") as f:
    #     texts = json.load(f)

    # return [x for x in texts if len(x) > 300]
    
    with open("notebooks/test_retriever.txt", "r") as f:
        texts = f.readlines()
    return texts


# ================================
# Create vectorstore for full text chunks
vectorstore = Chroma(collection_name="relevant_data", embedding_function=embedder)

# Загружаем тексты из S3
load_dotenv()
bucket_name = "palladium-md-to-chunks"
prefix = os.getenv("S3_PREFIX")

print("🔄 Загружаем данные из S3...")
texts = load_texts_from_s3(bucket_name, prefix)

if not texts:
    print("❌ Не удалось загрузить тексты из S3")
    exit(1)

# Convert text chunks to Document objects
documents = []
for i, text_chunk in enumerate(texts):
    # Create Document with metadata
    doc = Document(
        page_content=text_chunk,
        metadata={
            "chunk_id": i,
            "source": f"s3://{bucket_name}/{prefix or ''}"
        }
    )
    documents.append(doc)

print(f"📚 Создано документов: {len(documents)}")

# Add documents to vectorstore in batches
print("🔄 Добавляем документы в векторное хранилище...")

# ChromaDB has a batch size limit, so we'll add documents in smaller batches
batch_size = 1000  # Safe batch size for ChromaDB
total_docs = len(documents)

for i in range(0, total_docs, batch_size):
    batch = documents[i:i + batch_size]
    batch_num = i // batch_size + 1
    total_batches = (total_docs - 1) // batch_size + 1
    
    print(f"📝 Добавляем батч {batch_num}/{total_batches} ({len(batch)} документов)...")
    
    try:
        vectorstore.add_documents(batch)
        print(f"✅ Батч {batch_num} успешно добавлен")
    except Exception as e:
        print(f"❌ Ошибка при добавлении батча {batch_num}: {e}")
        continue

print("✅ Все документы успешно добавлены в Chroma")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ================================
# Тестирование ретривера
print("\n🔍 Тестируем ретривер...")

def test_retriever(query: str, k: int = 5):
    """Тестирует работу ретривера"""
    print(f"\n📝 Запрос: '{query}'")
    
    try:
        # Используем retriever для поиска
        retriever_test = vectorstore.as_retriever(search_kwargs={"k": k})
        results = retriever_test.get_relevant_documents(query)
        
        print(f"✅ Найдено {len(results)} релевантных документов:")
        print("=" * 70)
        
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Текст (первые 200 символов): {doc.page_content[:200]}...")
            print("-" * 70)
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании ретривера: {e}")

# Примеры запросов для тестирования
test_queries = [
    # "palladium refining",
    "вкусное мороженое"
    # "микроструктура металлов",
    # "mechanical properties"
]

# Тестируем с первым запросом
if test_queries:
    test_retriever(test_queries[0], k=3)
    
print(f"\n🎉 Векторное хранилище Chroma готово к использованию!")
print(f"📊 Всего проиндексировано документов: {len(documents)}")