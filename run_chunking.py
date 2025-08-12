#!/usr/bin/env python3
"""
Скрипт для чанкования маркдаун файлов на основе функционала из RAG репозитория.
Использует адаптированную версию TextSplitter для работы с маркдаун документами.
Поддерживает работу с S3 бакетами Yandex Object Storage.
"""

import json
import tiktoken
import boto3
import os
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import argparse
from datetime import datetime
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import tempfile

# Загрузка переменных окружения
load_dotenv()


class S3MarkdownChunker:
    """
    Класс для чанкования маркдаун документов с поддержкой S3 бакетов.
    Адаптирован на основе TextSplitter из src/text_splitter.py
    """
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50, encoding_name: str = "o200k_base"):
        """
        Инициализация чанкера.
        
        Args:
            chunk_size: Размер чанка в токенах
            chunk_overlap: Перекрытие между чанками в токенах
            encoding_name: Название кодировки для подсчета токенов
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        
        # Инициализация S3 клиента
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('S3_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY'),
            region_name='ru-central1'  # Для Yandex Cloud
        )
        
        self.source_bucket = "palladium-pdf-to-text"
        self.target_bucket = "palladium-md-to-chunks"
        
        # Проверяем существование бакетов
        self._ensure_bucket_exists(self.target_bucket)
    
    def _ensure_bucket_exists(self, bucket_name: str):
        """Проверяет существование бакета и создает его при необходимости."""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            print(f"Бакет {bucket_name} существует")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    self.s3_client.create_bucket(Bucket=bucket_name)
                    print(f"Создан бакет {bucket_name}")
                except ClientError as create_error:
                    print(f"Ошибка при создании бакета {bucket_name}: {create_error}")
                    raise
            else:
                print(f"Ошибка при проверке бакета {bucket_name}: {e}")
                raise
    
    def count_tokens(self, text: str) -> int:
        """Подсчет количества токенов в тексте."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    
    def _create_file_metadata(self, s3_key: str, content: str) -> Dict:
        """Создание метаданных для файла из S3."""
        file_hash = hashlib.sha1(content.encode('utf-8')).hexdigest()
        
        return {
            "file_name": Path(s3_key).name,
            "file_path": s3_key,
            "s3_key": s3_key,
            "sha1_hash": file_hash,
            "processed_at": datetime.now().isoformat(),
            "file_size_bytes": len(content.encode('utf-8'))
        }
    
    def _split_markdown_text(self, text: str) -> List[str]:
        """
        Разбиение маркдаун текста на чанки с использованием RecursiveCharacterTextSplitter.
        Адаптировано из _split_page метода.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n# ",      # Заголовки первого уровня
                "\n\n## ",     # Заголовки второго уровня
                "\n\n### ",    # Заголовки третьего уровня
                "\n\n",        # Параграфы
                "\n",          # Строки
                ". ",          # Предложения
                " ",           # Слова
                ""             # Символы
            ]
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    def _download_file_from_s3(self, s3_key: str) -> str:
        """Скачивание файла из S3 и возврат его содержимого."""
        try:
            response = self.s3_client.get_object(Bucket=self.source_bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            return content
        except ClientError as e:
            print(f"Ошибка при скачивании файла {s3_key}: {e}")
            raise
    
    def _upload_json_to_s3(self, json_data: Dict, s3_key: str):
        """Загрузка JSON данных в S3."""
        try:
            json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
            self.s3_client.put_object(
                Bucket=self.target_bucket,
                Key=s3_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"  Сохранено в S3: s3://{self.target_bucket}/{s3_key}")
        except ClientError as e:
            print(f"Ошибка при загрузке файла {s3_key}: {e}")
            raise
    
    def chunk_markdown_file_from_s3(self, s3_key: str) -> Dict:
        """
        Чанкование одного маркдаун файла из S3.
        
        Args:
            s3_key: Ключ файла в S3 бакете
            
        Returns:
            Словарь с метаданными файла и списком чанков
        """
        print(f"Обрабатываю файл: s3://{self.source_bucket}/{s3_key}")
        
        # Скачиваем содержимое файла
        content = self._download_file_from_s3(s3_key)
        
        # Создаем метаданные
        metadata = self._create_file_metadata(s3_key, content)
        
        # Разбиваем на чанки
        text_chunks = self._split_markdown_text(content)
        
        # Создаем структуру чанков с метаданными
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "id": i,
                "type": "content",
                "text": chunk_text.strip(),
                "length_tokens": self.count_tokens(chunk_text),
                "chunk_index": i,
                "total_chunks": len(text_chunks)
            }
            chunks.append(chunk)
        
        # Формируем результат в формате, совместимом с основным pipeline
        result = {
            "metainfo": metadata,
            "content": {
                "chunks": chunks,
                "original_text": content,
                "total_tokens": self.count_tokens(content),
                "total_chunks": len(chunks)
            }
        }
        
        print(f"  Создано {len(chunks)} чанков, общий размер: {self.count_tokens(content)} токенов")
        return result
    
    def list_markdown_files_in_s3(self, prefix: str = "") -> List[str]:
        """Получение списка всех маркдаун файлов в S3 бакете."""
        markdown_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        try:
            for page in paginator.paginate(Bucket=self.source_bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.md'):
                            markdown_files.append(key)
        except ClientError as e:
            print(f"Ошибка при получении списка файлов: {e}")
            raise
        
        return markdown_files
    
    def chunk_s3_bucket(self, prefix: str = "") -> None:
        """
        Чанкование всех маркдаун файлов в S3 бакете с сохранением структуры папок.
        
        Args:
            prefix: Префикс для фильтрации файлов в бакете
        """
        print(f"Поиск маркдаун файлов в бакете s3://{self.source_bucket}...")
        
        # Получаем список всех маркдаун файлов
        md_files = self.list_markdown_files_in_s3(prefix)
        
        if not md_files:
            print(f"Не найдено маркдаун файлов в бакете {self.source_bucket}")
            return
        
        print(f"Найдено {len(md_files)} файлов для обработки")
        
        processed_count = 0
        failed_count = 0
        
        for md_file_key in md_files:
            try:
                # Чанкуем файл
                chunked_data = self.chunk_markdown_file_from_s3(md_file_key)
                
                # Создаем путь для сохранения, сохраняя структуру папок
                # Меняем расширение с .md на .json
                output_key = md_file_key.replace('.md', '.json')
                
                # Загружаем результат в целевой бакет
                self._upload_json_to_s3(chunked_data, output_key)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Ошибка при обработке файла {md_file_key}: {e}")
                failed_count += 1
                continue
        
        print(f"\nОбработка завершена:")
        print(f"  Успешно обработано: {processed_count} файлов")
        print(f"  Ошибок: {failed_count} файлов")
        print(f"  Результаты сохранены в бакете: s3://{self.target_bucket}")


# Сохраняем старый класс для обратной совместимости
class MarkdownChunker:
    """
    Класс для чанкования маркдаун документов.
    Адаптирован на основе TextSplitter из src/text_splitter.py
    """
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50, encoding_name: str = "o200k_base"):
        """
        Инициализация чанкера.
        
        Args:
            chunk_size: Размер чанка в токенах
            chunk_overlap: Перекрытие между чанками в токенах
            encoding_name: Название кодировки для подсчета токенов
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
    
    def count_tokens(self, text: str) -> int:
        """Подсчет количества токенов в тексте."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    
    def _create_file_metadata(self, file_path: Path) -> Dict:
        """Создание метаданных для файла."""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha1(f.read()).hexdigest()
        
        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "sha1_hash": file_hash,
            "processed_at": datetime.now().isoformat(),
            "file_size_bytes": file_path.stat().st_size
        }
    
    def _split_markdown_text(self, text: str) -> List[str]:
        """
        Разбиение маркдаун текста на чанки с использованием RecursiveCharacterTextSplitter.
        Адаптировано из _split_page метода.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n# ",      # Заголовки первого уровня
                "\n\n## ",     # Заголовки второго уровня
                "\n\n### ",    # Заголовки третьего уровня
                "\n\n",        # Параграфы
                "\n",          # Строки
                ". ",          # Предложения
                " ",           # Слова
                ""             # Символы
            ]
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    def chunk_markdown_file(self, file_path: Path) -> Dict:
        """
        Чанкование одного маркдаун файла.
        
        Args:
            file_path: Путь к маркдаун файлу
            
        Returns:
            Словарь с метаданными файла и списком чанков
        """
        print(f"Обрабатываю файл: {file_path}")
        
        # Читаем содержимое файла
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Создаем метаданные
        metadata = self._create_file_metadata(file_path)
        
        # Разбиваем на чанки
        text_chunks = self._split_markdown_text(content)
        
        # Создаем структуру чанков с метаданными
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "id": i,
                "type": "content",
                "text": chunk_text.strip(),
                "length_tokens": self.count_tokens(chunk_text),
                "chunk_index": i,
                "total_chunks": len(text_chunks)
            }
            chunks.append(chunk)
        
        # Формируем результат в формате, совместимом с основным pipeline
        result = {
            "metainfo": metadata,
            "content": {
                "chunks": chunks,
                "original_text": content,
                "total_tokens": self.count_tokens(content),
                "total_chunks": len(chunks)
            }
        }
        
        print(f"  Создано {len(chunks)} чанков, общий размер: {self.count_tokens(content)} токенов")
        return result
    
    def chunk_directory(self, input_dir: Path, output_dir: Path, pattern: str = "*.md") -> None:
        """
        Чанкование всех маркдаун файлов в директории.
        
        Args:
            input_dir: Входная директория с маркдаун файлами
            output_dir: Выходная директория для сохранения чанков
            pattern: Паттерн для поиска файлов
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Создаем выходную директорию
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Находим все маркдаун файлы
        md_files = list(input_dir.rglob(pattern))
        
        if not md_files:
            print(f"Не найдено файлов по паттерну {pattern} в директории {input_dir}")
            return
        
        print(f"Найдено {len(md_files)} файлов для обработки")
        
        processed_count = 0
        failed_count = 0
        
        for md_file in md_files:
            try:
                # Чанкуем файл
                chunked_data = self.chunk_markdown_file(md_file)
                
                # Создаем имя выходного файла на основе хеша
                output_filename = f"{chunked_data['metainfo']['sha1_hash']}.json"
                output_path = output_dir / output_filename
                
                # Сохраняем результат
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chunked_data, f, indent=2, ensure_ascii=False)
                
                print(f"  Сохранено: {output_path}")
                processed_count += 1
                
            except Exception as e:
                print(f"Ошибка при обработке файла {md_file}: {e}")
                failed_count += 1
                continue
        
        print(f"\nОбработка завершена:")
        print(f"  Успешно обработано: {processed_count} файлов")
        print(f"  Ошибок: {failed_count} файлов")
        print(f"  Результаты сохранены в: {output_dir}")


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(description="Чанкование маркдаун файлов для RAG системы")
    
    # Добавляем выбор режима работы
    parser.add_argument("--mode", choices=['local', 's3'], default='s3', 
                       help="Режим работы: local (локальные файлы) или s3 (работа с S3 бакетами)")
    
    # Аргументы для локального режима
    parser.add_argument("--input-dir", type=str, help="Входная директория с маркдаун файлами (для локального режима)")
    parser.add_argument("--output-dir", type=str, help="Выходная директория для чанков (для локального режима)")
    parser.add_argument("--pattern", type=str, default="*.md", help="Паттерн для поиска файлов (по умолчанию: *.md)")
    
    # Аргументы для S3 режима
    parser.add_argument("--prefix", type=str, default="", help="Префикс для фильтрации файлов в S3 бакете")
    
    # Общие аргументы
    parser.add_argument("--chunk-size", type=int, default=300, help="Размер чанка в токенах (по умолчанию: 300)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Перекрытие между чанками в токенах (по умолчанию: 50)")
    parser.add_argument("--encoding", type=str, default="o200k_base", help="Кодировка для подсчета токенов (по умолчанию: o200k_base)")
    
    args = parser.parse_args()
    
    if args.mode == 's3':
        # Создаем S3 чанкер
        chunker = S3MarkdownChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            encoding_name=args.encoding
        )
        
        # Запускаем чанкование S3 бакета
        chunker.chunk_s3_bucket(prefix=args.prefix)
        
    else:  # local mode
        if not args.input_dir or not args.output_dir:
            print("Для локального режима необходимо указать --input-dir и --output-dir")
            return
        
        # Создаем обычный чанкер
        chunker = MarkdownChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            encoding_name=args.encoding
        )
        
        # Запускаем чанкование локальной директории
        chunker.chunk_directory(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            pattern=args.pattern
        )


def example_usage():
    """Пример использования для тестирования."""
    print("Примеры использования:")
    print("\n1. Работа с S3 бакетами (по умолчанию):")
    print("   python run_chunking.py --mode s3")
    print("   python run_chunking.py --mode s3 --prefix folder/subfolder/")
    
    print("\n2. Работа с локальными файлами:")
    print("   python run_chunking.py --mode local --input-dir notebooks/md_data --output-dir notebooks/chunked_data")
    
    print("\n3. Автоматический режим S3:")
    
    # Создаем S3 чанкер с параметрами по умолчанию
    try:
        chunker = S3MarkdownChunker(chunk_size=500, chunk_overlap=100)
        print("S3 чанкер инициализирован успешно")
        
        # Получаем список файлов для демонстрации
        files = chunker.list_markdown_files_in_s3()
        if files:
            print(f"Найдено {len(files)} маркдаун файлов в бакете {chunker.source_bucket}")
            print("Первые 5 файлов:")
            for f in files[:5]:
                print(f"  - {f}")
        else:
            print(f"Маркдаун файлы не найдены в бакете {chunker.source_bucket}")
            
    except Exception as e:
        print(f"Ошибка при инициализации S3 чанкера: {e}")
        print("Проверьте настройки S3 в .env файле")


if __name__ == "__main__":
    # Если запускается как скрипт, проверяем аргументы
    import sys
    
    if len(sys.argv) > 1:
        # Запуск с аргументами командной строки
        main()
    else:
        # Запуск примера использования
        print("Запуск в режиме примера использования...")
        print("Для использования с аргументами командной строки:")
        print("python run_chunking.py --mode s3 [--prefix folder/]")
        print("python run_chunking.py --mode local --input-dir <input_dir> --output-dir <output_dir>")
        print()
        example_usage()
