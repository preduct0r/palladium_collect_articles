#!/usr/bin/env python3
"""
Скрипт для извлечения всех DOI из файла dois.csv 
и вывода их в виде строки, разделенной пробелами.
"""

import pandas as pd
import os

def extract_dois():
    """Извлекает DOI из CSV файла и возвращает строку с DOI, разделенными пробелами."""
    
    # Путь к файлу CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'dois.csv')
    
    try:
        # Читаем CSV файл
        df = pd.read_csv(csv_path)
        
        # Извлекаем колонку с DOI
        dois = df['doi'].dropna().unique()  # Убираем NaN значения и дубликаты
        
        # Объединяем все DOI в строку, разделенную пробелами
        doi_string = ' '.join(dois)
        
        return doi_string
        
    except FileNotFoundError:
        print(f"Ошибка: файл {csv_path} не найден")
        return ""
    except KeyError:
        print("Ошибка: колонка 'doi' не найдена в CSV файле")
        return ""
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return ""

def main():
    """Основная функция."""
    doi_string = extract_dois()
    if doi_string:
        print(doi_string)
    
if __name__ == "__main__":
    main()
