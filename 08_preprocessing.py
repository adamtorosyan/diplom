import logging
import re  # Импортируем модуль для работы с регулярными выражениями
from concurrent.futures import (
    ProcessPoolExecutor,  # Импортируем для многопоточной обработки
)

import numpy as np
import pandas as pd  # Импортируем для работы с данными в формате DataFrame
import pymorphy2  # Импортируем для лемматизации на русском языке
import stop_words  # Импортируем для работы со стоп-словами


def setup_logging():
    # Настраиваем логирование для вывода в консоль и файл
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("preprocessing.log")],
    )


def keep_only_rus(text):
    # Функция удаляет все символы, кроме русских букв, цифр, точек и пробелов
    return re.sub(r"[^А-Яа-я0-9:. ]", " ", text)


def del_double_spaces(text):
    # Функция заменяет множественные пробелы на один и удаляет пробелы в начале и конце строки
    return re.sub(r"\s+", " ", text).strip()


# Загружаем стоп-слова один раз, чтобы не загружать их в каждой функции
STOPWORDS = set(stop_words.get_stop_words("russian")).union(
    stop_words.get_stop_words("english")
)


def del_stopwords(text):
    # Функция удаляет стоп-слова из текста
    return " ".join(
        [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    )


def lemmatize(raw_text):
    # Инициализируем морфологический анализатор
    morph = pymorphy2.MorphAnalyzer()
    words = raw_text.split()
    # Преобразуем слова в их начальную форму
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmatized_words)


def process_text(data_chunk):
    # Функция выполняет последовательную обработку текста
    data_chunk["text_prep"] = data_chunk["text_prep"].apply(keep_only_rus)
    data_chunk["text_prep"] = data_chunk["text_prep"].apply(del_double_spaces)
    data_chunk["text_prep"] = data_chunk["text_prep"].apply(lemmatize)
    data_chunk["text_prep"] = data_chunk["text_prep"].apply(del_stopwords)
    data_chunk["text_prep"] = data_chunk["text_prep"].apply(del_double_spaces)
    return data_chunk


def main():
    try:
        # Загружаем данные из CSV файла
        data = pd.read_csv(
            r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\extracted_cases_marked.csv",
            index_col=0,
        )
        logging.info("Starting preprocessing...")

        # Приводим текст к нижнему регистру
        data["text_prep"] = data["text"].str.lower()
        logging.info("Text converted to lowercase.")

        # Определяем количество частей для разбиения данных и количество процессов, исходя из технических характеристик устройства
        num_partitions = 4
        data_split = np.array_split(data, num_partitions)
        with ProcessPoolExecutor(max_workers=num_partitions) as executor:
            # Обрабатываем данные параллельно
            data = pd.concat(executor.map(process_text, data_split))

        # Удаляем строки с пустыми значениями в колонке 'text_prep'
        data = data[data["text_prep"].notna()]
        logging.info("Preprocessing completed.")

        # Сохраняем предобработанные данные в CSV файл
        data.to_csv("extracted_cases_preprocessed.csv")
        logging.info("Preprocessed data saved to 'extracted_cases_preprocessed.csv'.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Настраиваем логирование и запускаем основной процесс
    setup_logging()
    main()
