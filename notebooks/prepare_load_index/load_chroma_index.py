#!/usr/bin/env python3
"""
Utility script for loading and using Chroma index on another machine.

Usage:
    # Load from archive
    python load_chroma_index.py --from-archive chroma_index.tar.gz
    
    # Load existing index
    python load_chroma_index.py --index-path ./chroma_db
    
    # Interactive query mode
    python load_chroma_index.py --index-path ./chroma_db --interactive
"""

import argparse
import json
import os
import sys
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Import functions from the main script
sys.path.append(os.path.dirname(__file__))
from chroma_index import extract_index_from_package, load_index_info, test_retriever

class ChromaIndexLoader:
    def __init__(self, embedding_model: str = "Qwen/Qwen3-Embedding-8B", 
                 base_url: str = "http://localhost:8090/v1",
                 api_key: str = "EMPTY"):
        """Initialize the index loader with embedding configuration"""
        self.embedder = OpenAIEmbeddings(
            model=embedding_model,
            base_url=base_url,
            api_key=api_key
        )
        self.vectorstore = None
        
    def load_from_archive(self, archive_path: str, target_directory: str = "./chroma_db") -> bool:
        """Load index from archive file"""
        try:
            extract_index_from_package(archive_path, target_directory)
            return self.load_from_directory(target_directory)
        except Exception as e:
            print(f"❌ Ошибка при загрузке из архива: {e}")
            return False
    
    def load_from_directory(self, index_path: str) -> bool:
        """Load index from directory"""
        if not os.path.exists(index_path):
            print(f"❌ Директория индекса не найдена: {index_path}")
            return False
            
        try:
            # Load index info
            info = load_index_info(index_path)
            if info:
                print(f"📋 Информация об индексе:")
                for key, value in info.items():
                    print(f"   {key}: {value}")
                collection_name = info.get("collection_name", "relevant_data")
            else:
                print("⚠️ Информация об индексе не найдена, используем параметры по умолчанию")
                collection_name = "relevant_data"
            
            # Load vectorstore
            print(f"🔄 Загружаем векторное хранилище из {index_path}")
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedder,
                persist_directory=index_path
            )
            
            # Get document count
            collection = self.vectorstore._collection
            count = collection.count()
            print(f"✅ Успешно загружено {count} документов")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке индекса: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> list:
        """Search for relevant documents"""
        if not self.vectorstore:
            print("❌ Индекс не загружен")
            return []
            
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            results = retriever.get_relevant_documents(query)
            return results
        except Exception as e:
            print(f"❌ Ошибка при поиске: {e}")
            return []
    
    def interactive_mode(self):
        """Interactive query mode"""
        if not self.vectorstore:
            print("❌ Индекс не загружен")
            return
            
        print("\n🔍 Интерактивный режим поиска")
        print("Введите запрос для поиска (или 'quit' для выхода):")
        
        while True:
            try:
                query = input("\n>>> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 До свидания!")
                    break
                    
                if not query:
                    continue
                    
                # Ask for number of results
                try:
                    k_input = input("Количество результатов (по умолчанию 5): ").strip()
                    k = int(k_input) if k_input else 5
                except ValueError:
                    k = 5
                
                # Perform search
                test_retriever(self.vectorstore, query, k)
                
            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")

def main():
    parser = argparse.ArgumentParser(description="Load and use Chroma vector index")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-archive", metavar="PATH", 
                      help="Load index from archive file")
    group.add_argument("--index-path", metavar="PATH", 
                      help="Load index from directory")
    
    parser.add_argument("--target-dir", metavar="PATH", default="./chroma_db",
                       help="Target directory for extracted index (default: ./chroma_db)")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive query mode")
    parser.add_argument("--query", metavar="TEXT",
                       help="Single query to execute")
    parser.add_argument("--results", type=int, default=5,
                       help="Number of results to return (default: 5)")
    
    # Embedding model configuration
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-8B",
                       help="Embedding model name")
    parser.add_argument("--base-url", default="http://localhost:8090/v1",
                       help="Base URL for embedding API")
    parser.add_argument("--api-key", default="EMPTY",
                       help="API key for embedding service")
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = ChromaIndexLoader(
        embedding_model=args.embedding_model,
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    # Load index
    success = False
    if args.from_archive:
        success = loader.load_from_archive(args.from_archive, args.target_dir)
    elif args.index_path:
        success = loader.load_from_directory(args.index_path)
    
    if not success:
        print("❌ Не удалось загрузить индекс")
        sys.exit(1)
    
    # Execute query or start interactive mode
    if args.query:
        print(f"\n🔍 Выполняем запрос: '{args.query}'")
        test_retriever(loader.vectorstore, args.query, args.results)
    elif args.interactive:
        loader.interactive_mode()
    else:
        print("\n✅ Индекс успешно загружен!")
        print("💡 Используйте --interactive для интерактивного режима или --query для выполнения запроса")

if __name__ == "__main__":
    main() 