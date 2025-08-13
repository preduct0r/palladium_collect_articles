import asyncio
import json
import os
import shutil
import tarfile
from typing import List, Dict, Optional
import boto3
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document

# Configuration
PERSIST_DIRECTORY = "/home/ubuntu/kotov_projects/palladium_collect_articles/chroma_db"
COLLECTION_NAME = "relevant_data"

# Инициализация эмбеддера
embedder = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B", 
    base_url="http://localhost:8090/v1",
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
    """Загружает тексты из JSON файла"""
    
    json_file_path = "/home/ubuntu/kotov_projects/texts.json"
    
    print(f"📦 Чтение текстов из файла: {json_file_path}")
    
    if not os.path.exists(json_file_path):
        print(f"❌ Файл {json_file_path} не найден")
        return []
    
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            texts = json.load(f)
        
        print(f"📄 Загружено текстов из файла: {len(texts)}")
        
        # Фильтруем тексты по длине (больше 300 символов)
        filtered_texts = [text for text in texts if isinstance(text, str) and len(text) > 300]
        
        print(f"✅ После фильтрации (>300 символов): {len(filtered_texts)} текстов")
        
        return filtered_texts
        
    except Exception as e:
        print(f"❌ Ошибка при чтении файла {json_file_path}: {e}")
        return []


def create_or_load_vectorstore(persist_directory: str = PERSIST_DIRECTORY, 
                               collection_name: str = COLLECTION_NAME) -> Chroma:
    """Создает новое или загружает существующее векторное хранилище"""
    
    # Создаем директорию если её нет
    os.makedirs(persist_directory, exist_ok=True)
    
    # Проверяем, существует ли уже индекс
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"📂 Найдено существующее векторное хранилище в {persist_directory}")
        print("🔄 Загружаем существующий индекс...")
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=persist_directory
        )
        
        # Получаем количество документов в существующем индексе
        collection = vectorstore._collection
        count = collection.count()
        print(f"📊 Загружено {count} документов из существующего индекса")
        
        return vectorstore
    else:
        print(f"🆕 Создаем новое векторное хранилище в {persist_directory}")
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=persist_directory
        )
        
        return vectorstore

def save_index_info(persist_directory: str, metadata: Dict):
    """Сохраняет метаданные об индексе"""
    info_file = os.path.join(persist_directory, "index_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Информация об индексе сохранена в {info_file}")

def load_index_info(persist_directory: str) -> Optional[Dict]:
    """Загружает метаданные об индексе"""
    info_file = os.path.join(persist_directory, "index_info.json")
    if os.path.exists(info_file):
        with open(info_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def package_index_for_transfer(persist_directory: str, 
                               output_path: str = "chroma_index.tar.gz") -> str:
    """Упаковывает индекс в архив для передачи на другую машину"""
    print(f"📦 Упаковываем индекс из {persist_directory} в {output_path}")
    
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Директория {persist_directory} не найдена")
    
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(persist_directory, arcname="chroma_db")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"✅ Индекс упакован в {output_path} (размер: {file_size:.2f} MB)")
    
    return output_path

def extract_index_from_package(archive_path: str, 
                               target_directory: str = PERSIST_DIRECTORY) -> str:
    """Распаковывает индекс из архива"""
    print(f"📦 Распаковываем индекс из {archive_path} в {target_directory}")
    
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Архив {archive_path} не найден")
    
    # Удаляем существующую директорию если она есть
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    
    # Создаем родительскую директорию
    parent_dir = os.path.dirname(target_directory)
    os.makedirs(parent_dir, exist_ok=True)
    
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=parent_dir)
    
    print(f"✅ Индекс распакован в {target_directory}")
    
    # Загружаем информацию об индексе
    info = load_index_info(target_directory)
    if info:
        print(f"📋 Информация об индексе:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    return target_directory

def test_retriever(vectorstore: Chroma, query: str, k: int = 5):
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

def main():
    """Основная функция для создания/загрузки индекса"""
    
    # ================================
    # Create or load vectorstore
    
    vectorstore = create_or_load_vectorstore()
    
    # Проверяем, нужно ли добавлять новые документы
    collection = vectorstore._collection
    existing_count = collection.count()
    
    if existing_count == 0:
        print("📝 Индекс пустой, добавляем документы...")
        
        # Загружаем тексты из JSON файла
        load_dotenv()
        bucket_name = "palladium-md-to-chunks"  # Не используется, но оставляем для совместимости
        prefix = os.getenv("S3_PREFIX")  # Не используется, но оставляем для совместимости

        print("🔄 Загружаем данные из JSON файла...")
        texts = load_texts_from_s3(bucket_name, prefix)

        if not texts:
            print("❌ Не удалось загрузить тексты из JSON файла")
            exit(1)

        # Convert text chunks to Document objects
        documents = []
        for i, text_chunk in enumerate(texts):
            # Create Document with metadata
            doc = Document(
                page_content=text_chunk,
                metadata={
                    "chunk_id": i,
                    "source": "texts.json",
                    "text_length": len(text_chunk)
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
        
        # Сохраняем метаданные об индексе
        import datetime
        metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "total_documents": len(documents),
            "embedding_model": "Qwen/Qwen3-Embedding-8B",
            "collection_name": COLLECTION_NAME,
            "source_file": "texts.json",
            "filtered_min_length": 300
        }
        save_index_info(PERSIST_DIRECTORY, metadata)
        
    else:
        print(f"📊 Использую существующий индекс с {existing_count} документами")

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # ================================
    # Тестирование ретривера
    print("\n🔍 Тестируем ретривер...")

    # Примеры запросов для тестирования
    test_queries = [
        # "palladium refining",
        "chocolate ice cream"
        # "микроструктура металлов",
        # "mechanical properties"
    ]

    # Тестируем с первым запросом
    if test_queries:
        test_retriever(vectorstore, test_queries[0], k=30)
        
    print(f"\n🎉 Векторное хранилище Chroma готово к использованию!")
    print(f"📊 Всего проиндексировано документов: {existing_count}")
    print(f"💾 Индекс сохранен в: {PERSIST_DIRECTORY}")
    
    # Предлагаем упаковать индекс для передачи
    response = input("\n🤔 Хотите упаковать индекс для передачи на другую машину? (y/n): ").strip().lower()
    if response == 'y':
        output_path = "chroma_index.tar.gz"  # Устанавливаем значение по умолчанию
        # output_path = input("📁 Введите путь для архива (по умолчанию: chroma_index.tar.gz): ").strip()
        # if not output_path:
        #     output_path = "chroma_index.tar.gz"
        
        try:
            package_index_for_transfer(PERSIST_DIRECTORY, output_path)
            print(f"✅ Готово! Теперь вы можете передать файл {output_path} на другую машину")
            print("📋 Для восстановления на другой машине используйте функцию extract_index_from_package()")
        except Exception as e:
            print(f"❌ Ошибка при упаковке: {e}")

if __name__ == "__main__":
    main()