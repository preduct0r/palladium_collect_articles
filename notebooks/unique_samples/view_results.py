#!/usr/bin/env python3
"""
Скрипт для просмотра результатов анализа сходства
"""
import pandas as pd
import argparse

def view_results(file_path: str, num_results: int = 10):
    """
    Просмотр результатов из Excel файла
    
    Args:
        file_path: Путь к файлу с результатами
        num_results: Количество результатов для показа
    """
    try:
        df = pd.read_excel(file_path)
        print(f"\n📊 Результаты из файла: {file_path}")
        print(f"📈 Всего найдено: {len(df)} пар")
        print("=" * 80)
        
        for i, row in df.head(num_results).iterrows():
            print(f"\n{i+1}. Сходство: {row['Сходство']:.4f}")
            print(f"   Индексы: {row['Индекс_1']} ↔ {row['Индекс_2']}")
            print(f"   Вопрос 1: {row['Вопрос_1'][:100]}...")
            print(f"   Вопрос 2: {row['Вопрос_2'][:100]}...")
            
        if len(df) > num_results:
            print(f"\n... и еще {len(df) - num_results} пар")
            
    except Exception as e:
        print(f"❌ Ошибка при чтении файла {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Просмотр результатов анализа сходства")
    parser.add_argument("--file", default="top_similar_pairs.xlsx", help="Файл с результатами")
    parser.add_argument("--num", type=int, default=10, help="Количество результатов для показа")
    parser.add_argument("--all", action="store_true", help="Показать результаты из всех файлов")
    
    args = parser.parse_args()
    
    if args.all:
        print("🔍 Просматриваю все результаты...")
        
        # Топ похожие пары
        print("\n" + "="*80)
        print("🏆 ТОП НАИБОЛЕЕ ПОХОЖИХ ПАР")
        print("="*80)
        view_results("top_similar_pairs.xlsx", args.num)
        
        # Потенциальные дубликаты
        print("\n" + "="*80)
        print("🔄 ПОТЕНЦИАЛЬНЫЕ ДУБЛИКАТЫ")
        print("="*80)
        view_results("potential_duplicates.xlsx", args.num)
    else:
        view_results(args.file, args.num)

if __name__ == "__main__":
    main() 