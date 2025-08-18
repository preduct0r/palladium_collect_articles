import pandas as pd
from collections import Counter
from pathlib import Path

# Путь к файлу
file_path = "/home/den/Documents/Nornikel/1_линия_поддержки/Super Маршрутная карта 15.08.2025.xlsx"

# Загружаем Excel-файл
xls = pd.ExcelFile(file_path)

# Словарь для хранения названий столбцов
all_columns = []

# Проходим по всем листам и собираем столбцы
for sheet in xls.sheet_names:
    try:
        # Загружаем данные с каждого листа
        df = pd.read_excel(file_path, sheet_name=sheet)
        
        # Добавляем названия столбцов в общий список
        all_columns.extend(df.columns.tolist())
    except Exception as e:
        print(f"Ошибка при загрузке листа {sheet}: {e}")

# Подсчитываем частоту встречаемости столбцов
column_counts = Counter(all_columns)

# Выводим результат
print("Частота встречаемости столбцов:")
for column, count in column_counts.items():
    print(f"{column}: {count}")

# Сохраняем результат в текстовый файл рядом со скриптом
output_file = Path(__file__).resolve().parent / "column_counts.txt"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Частота встречаемости столбцов:\n")
        for column, count in column_counts.items():
            f.write(f"{column}: {count}\n")
    print(f"Результат сохранён в файл: {output_file}")
except Exception as e:
    print(f"Ошибка при сохранении результата: {e}")
