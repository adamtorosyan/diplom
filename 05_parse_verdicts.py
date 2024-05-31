import gzip  # Импорт модуля для работы с gzip-архивами
import os  # Импорт модуля для работы с операционной системой
import re  # Импорт модуля для работы с регулярными выражениями
import shutil  # Импорт модуля для работы с файлами и директориями

import pandas as pd  # Импорт библиотеки pandas для работы с таблицами данных

# Чтение CSV файла в DataFrame
hom = pd.read_csv(
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\2011_homicide.csv"
)

# Добавление колонки с идентификаторами
hom["id"] = range(1, len(hom) + 1)

# Создание директории для хранения выборки приговоров
sample_dir = "C:/Users/WarSa/OneDrive/Рабочий стол/диплом/rospravosudie_sou/verdicts/"
os.makedirs(sample_dir, exist_ok=True)  # Создание директории, если она не существует

# Обработка каждой строки в выборке
for i, row in hom.iterrows():
    # Копирование файла в директорию выборки
    source_file = row["path"]  # Исходный файл
    dest_file = os.path.join(sample_dir, os.path.basename(source_file))  # Путь назначения
    shutil.copyfile(source_file, dest_file)  # Копирование файла

    # Распаковка скопированного файла
    with gzip.open(dest_file, "rb") as f_in:  # Открытие сжатого файла
        with open(dest_file[:-3], "wb") as f_out:  # Создание файла для распакованного содержимого
            shutil.copyfileobj(f_in, f_out)  # Копирование содержимого

    # Удаление сжатого файла
    os.remove(dest_file)

    # Чтение распакованного XML файла
    with open(dest_file[:-3], "r", encoding="utf-8") as xml_file:
        verdict_raw_text = xml_file.read()  # Чтение содержимого файла

    # Извлечение текста приговора из содержимого XML
    match = re.search(r"CDATA\[(.*?)\]", verdict_raw_text, re.DOTALL)  # Поиск текста внутри CDATA
    verdict_body = match.group(1) if match else ""  # Получение текста приговора

    # Создание HTML содержимого
    verdict_html = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>№{i + 1} ID {row['id']}</title>
  </head>
  <body>
    {verdict_body}
  </body>
</html>"""

    # Запись HTML содержимого в файл
    html_file = os.path.join(
        sample_dir, f"{i + 1}.html"
    )  # Формирование пути к HTML файлу с индексом, начинающимся с 1
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(verdict_html)  # Запись HTML содержимого

    # Удаление распакованного XML файла
    os.remove(dest_file[:-3])
