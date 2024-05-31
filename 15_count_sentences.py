# Импортирование необходимых библиотек
import logging
import statistics
from multiprocessing import Pool, cpu_count

import pandas as pd
import spacy

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Загрузка русской модели языка Spacy глобально
nlp = spacy.load("ru_core_news_sm")


# Функция для обработки частей данных (для работы с многопроцессорностью)
def process_texts(texts):
    sentence_counts = []  # Список для хранения количества предложений
    for text in texts:
        doc = nlp(text)  # Применение модели Spacy к тексту
        sentence_count = len(list(doc.sents))  # Подсчет количества предложений
        sentence_counts.append(
            sentence_count
        )  # Добавление количества предложений в список
    return sentence_counts  # Возвращение списка с количеством предложений


def main(df):
    # Определение оптимального количества процессов на основе количества ядер процессора
    num_processes = 15

    # Создание частей текстов из DataFrame для многопроцессорности
    chunk_size = len(df) // num_processes
    text_chunks = [
        df["text"].iloc[i : i + chunk_size].tolist()
        for i in range(0, len(df), chunk_size)
    ]
    text_prep_chunks = [
        df["text_prep"].iloc[i : i + chunk_size].tolist()
        for i in range(0, len(df), chunk_size)
    ]

    # Параллельная обработка столбца 'text'
    with Pool(processes=num_processes) as pool:
        results_text = pool.map(process_texts, text_chunks)

    # Параллельная обработка столбца 'text_prep'
    with Pool(processes=num_processes) as pool:
        results_text_prep = pool.map(process_texts, text_prep_chunks)

    # Объединение списка с количеством предложений
    counts_text = [count for sublist in results_text for count in sublist]
    counts_text_prep = [count for sublist in results_text_prep for count in sublist]

    # Вычисление общего количества, среднего и медианного количества предложений для каждого столбца
    total_sentences_text = sum(counts_text)
    mean_sentences_text = statistics.mean(counts_text)
    median_sentences_text = statistics.median(counts_text)

    total_sentences_text_prep = sum(counts_text_prep)
    mean_sentences_text_prep = statistics.mean(counts_text_prep)
    median_sentences_text_prep = statistics.median(counts_text_prep)

    # Печать результатов для обоих столбцов
    print("Text column - Total count of sentences:", total_sentences_text)
    print("Text column - Mean count of sentences:", mean_sentences_text)
    print("Text column - Median count of sentences:", median_sentences_text)

    print("Text Prep column - Total count of sentences:", total_sentences_text_prep)
    print("Text Prep column - Mean count of sentences:", mean_sentences_text_prep)
    print("Text Prep column - Median count of sentences:", median_sentences_text_prep)


if __name__ == "__main__":
    # Загрузка данных из CSV файла
    df = pd.read_csv("extracted_cases_preprocessed.csv")
    main(df)
