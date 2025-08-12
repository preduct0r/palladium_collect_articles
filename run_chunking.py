#!/usr/bin/env python3
"""
Скрипт для чанкования маркдаун файлов на основе функционала из RAG репозитория.
Использует адаптированную версию TextSplitter для работы с маркдаун документами.
"""

import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import argparse
from datetime import datetime


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
    
    parser.add_argument("input_dir", type=str, help="Входная директория с маркдаун файлами")
    parser.add_argument("output_dir", type=str, help="Выходная директория для чанков")
    parser.add_argument("--chunk-size", type=int, default=300, help="Размер чанка в токенах (по умолчанию: 300)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Перекрытие между чанками в токенах (по умолчанию: 50)")
    parser.add_argument("--pattern", type=str, default="*.md", help="Паттерн для поиска файлов (по умолчанию: *.md)")
    parser.add_argument("--encoding", type=str, default="o200k_base", help="Кодировка для подсчета токенов (по умолчанию: o200k_base)")
    
    args = parser.parse_args()
    
    # Создаем чанкер
    chunker = MarkdownChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        encoding_name=args.encoding
    )
    
    # Запускаем чанкование
    chunker.chunk_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        pattern=args.pattern
    )


def example_usage():
    """Пример использования для тестирования."""
    # Создаем чанкер с параметрами по умолчанию
    chunker = MarkdownChunker(chunk_size=500, chunk_overlap=100)
    
    # Пример чанкования одного файла
    # result = chunker.chunk_markdown_file(Path("example.md"))
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Пример чанкования директории
    input_dir = Path("notebooks/md_data")
    output_dir = Path("notebooks/chunked_data")
    
    if input_dir.exists():
        print("Запуск чанкования директории...")
        chunker.chunk_directory(input_dir, output_dir)
    else:
        print(f"Директория {input_dir} не найдена. Создайте её или измените путь.")


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
        print("python run_chunker.py <input_dir> <output_dir> [--chunk-size 300] [--chunk-overlap 50]")
        print()
        example_usage()
