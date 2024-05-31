import os

import pandas as pd

# Путь к директории, содержащей текстовые файлы
directory = (
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\extracted_texts"
)
output_csv = (
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\extracted_cases.csv"
)

# Инициализация пустого списка для хранения данных
data = []

# Проверка, существует ли выходной CSV файл
if os.path.exists(output_csv):
    # Если файл существует, загрузить его и получить существующие ID
    extracted_cases = pd.read_csv(output_csv)
    existing_ids = set(extracted_cases["ID"])
else:
    # Если файл не существует, создать новый DataFrame и пустой набор ID
    extracted_cases = pd.DataFrame(columns=["ID", "text"])
    existing_ids = set()

# Итерация по каждому файлу в директории
for filename in os.listdir(directory):
    # Проверка, что файл имеет расширение .txt и его ID (имя файла) не существует в выходном CSV
    if filename.endswith(".txt") and filename not in existing_ids:
        # Чтение содержимого текстового файла
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            text = file.read()

        # Добавление ID файла и содержимого текста в список данных
        data.append({"ID": filename, "text": text})
    else:
        # Пропуск файла, если его ID уже существует в выходном CSV
        print(f"Skipping {filename} as it already exists in the output CSV.")

# Создание DataFrame из нового списка данных
new_data_df = pd.DataFrame(data)

# Добавление новых данных в существующий DataFrame
if not new_data_df.empty:
    extracted_cases = pd.concat([extracted_cases, new_data_df], ignore_index=True)

# Сохранение обновленного DataFrame в CSV файл
extracted_cases.to_csv(output_csv, index=False)

print(f"DataFrame has been saved to '{output_csv}'")
