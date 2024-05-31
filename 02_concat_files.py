import logging  # Импорт модуля для логирования
import os  # Импорт модуля для работы с операционной системой

import pandas as pd  # Импорт модуля pandas для работы с таблицами

# Настройка логирования: создание файла для записи логов и указание формата сообщений
logging.basicConfig(
    filename="concatenation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Директория, содержащая CSV файлы
directory = "/home/adam/diplom/диплом/rospravosudie_sou/"

# Список всех CSV файлов в директории
csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]

# Если CSV файлы не найдены, запись предупреждения в лог и выход из программы
if not csv_files:
    logging.warning("No CSV files found in the directory.")
    exit()

# Имя выходного файла для объединенных данных
output_file = "/home/adam/diplom/диплом/rospravosudie_sou/rps_dataset.csv"

# Открытие выходного файла в режиме добавления
with open(output_file, "a") as f_out:
    # Перебор каждого CSV файла
    for i, file in enumerate(csv_files, 1):
        logging.info(f"Processing file {i}/{len(csv_files)}: {file}")
        # Открытие каждого CSV файла в режиме чтения
        with open(os.path.join(directory, file), "r") as f_in:
            # Копирование содержимого из входного файла в выходной файл
            for line in f_in:
                f_out.write(line)
        # Удаление промежуточного файла после обработки
        os.remove(os.path.join(directory, file))

logging.info(f"Concatenated CSV files into {output_file}")
print("Concatenation complete. rps_dataset.csv has been created.")
