#!/usr/bin/env python3
"""
Простой скрипт для запуска анализа сходства вопросов
"""
import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from find_similar_pairs import SimilarityFinder

def main():
    print("🔍 Запускаю анализ сходства вопросов...")
    print("=" * 50)
    
    # Путь к файлу с данными
    excel_file = "BENCH2.xlsx"
    
    # Создаем анализатор
    finder = SimilarityFinder(
        excel_file_path=excel_file,
        cache_dir="./cache"
    )
    
    # Запускаем анализ
    try:
        top_pairs, duplicates = finder.run_analysis(
            top_k=15,  # Топ-15 наиболее похожих пар
            similarity_threshold=0.90,  # Порог для дубликатов 90%
            force_recompute=False  # Используем кэш если есть
        )
        
        print("\n✅ Анализ завершен успешно!")
        print(f"📊 Найдено {len(top_pairs)} топ похожих пар")
        print(f"🔄 Найдено {len(duplicates)} потенциальных дубликатов")
        print("\n📁 Результаты сохранены в файлы:")
        print("   - top_similar_pairs.xlsx")
        if duplicates:
            print("   - potential_duplicates.xlsx")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 