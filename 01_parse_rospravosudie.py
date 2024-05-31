import gzip  # Импорт модуля для работы с gzip-архивами
import logging  # Импорт модуля для логирования

import pandas as pd  # Импорт модуля pandas для работы с таблицами
from bs4 import BeautifulSoup  # Импорт BeautifulSoup для парсинга HTML

# Определение списка колонок, которые будут использоваться в DataFrame
columns_list = [
    "region",  # Регион
    "court",  # Суд
    "judge",  # Судья
    "vidpr",  # Вид преступления
    "etapd",  # Этап дела
    "category",  # Категория дела
    "result",  # Результат дела
    "date",  # Дата
    "url",  # URL документа
    "vid_dokumenta",  # Вид документа
    "path",  # Путь к файлу
]

# Настройка логирования: создание файла для записи логов и указание формата сообщений
logging.basicConfig(
    filename="processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Функция для обработки файлов и генерации строк для DataFrame
def process_files(files):
    for counter, file in enumerate(
        files, start=1
    ):  # Перебор файлов с их нумерацией, начиная с 1
        file = base_dir + file  # Формирование полного пути к файлу
        with gzip.open(
            file, "rb"
        ) as f:  # Открытие файла в режиме чтения бинарных данных
            try:
                file_content = f.read().decode(
                    "utf-8"
                )  # Чтение содержимого файла и декодирование в строку
            except UnicodeDecodeError as e:  # Обработка ошибок декодирования
                logging.error(
                    f"Ошибка декодирования файла '{file}': {e}"
                )  # Запись ошибки в лог
                continue  # Переход к следующему файлу

            soup = BeautifulSoup(
                file_content, "html.parser"
            )  # Парсинг HTML содержимого файла

            # Создание словаря со значениями тегов для каждой колонки
            keys = columns_list  # Список ключей (имен колонок)
            values = [
                tag.text for tag in soup.find_all(keys)
            ]  # Получение текстовых значений для каждого тега
            row = dict(
                zip(keys, values)
            )  # Создание словаря, сопоставляя ключи и значения
            row["path"] = file  # Добавление пути к файлу в словарь

            yield row  # Генерация строки для DataFrame

        if counter % 10000 == 0:  # Логирование каждых 10000 обработанных файлов
            logging.info(f"Обработано {counter} файлов.")  # Запись информации в лог


# Чтение списка имен файлов из указанного текстового файла и получение 5-го столбца (имена файлов)
df_files = pd.read_csv(
    "/Users/WarSa/OneDrive/Рабочий стол/диплом/rospravosudie_sou/sou.txt",
    delim_whitespace=True,  # Указание разделителя
    header=None,  # Отсутствие заголовков
    nrows=100000,  # Чтение первых 100000 строк
)[
    5
]  # Получение 5-го столбца с именами файлов
base_dir = "/Users/WarSa/OneDrive/Рабочий стол/диплом/rospravosudie_sou/sou/"  # Базовый каталог с файлами

# Создание пустого DataFrame с указанными колонками
df = pd.DataFrame(columns=columns_list)

# Обработка файлов и сбор результатов в список
chunk_size = 10000  # Размер чанка (количество строк в одном чанке)
processed_chunks = []  # Список для хранения обработанных фрагментов DataFrame
for chunk in pd.read_csv(
    "/Users/WarSa/OneDrive/Рабочий стол/диплом/rospravosudie_sou/sou.txt",
    delim_whitespace=True,  # Указание разделителя
    header=None,  # Отсутствие заголовков
    chunksize=chunk_size,  # Размер чанка
    nrows=100000,  # Чтение первых 100000 строк
):
    chunk_generator = process_files(
        chunk[5]
    )  # Создание генератора строк для текущего чанка
    processed_chunks.append(
        pd.DataFrame(chunk_generator)
    )  # Преобразование генератора в DataFrame и добавление в список

# Объединение всех обработанных фрагментов в один DataFrame
df = pd.concat(processed_chunks, ignore_index=True)

# Логирование завершения обработки
logging.info("Обработка завершена.")

# Сохранение итогового DataFrame в CSV файл
df.to_csv(
    "/Users/WarSa/OneDrive/Рабочий стол/диплом/rospravosudie_sou/rps_dataset.csv",
    index=False,  # Отключение сохранения индексов
)
