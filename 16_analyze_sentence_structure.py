import logging
from collections import Counter
from multiprocessing import Pool, cpu_count

import pandas as pd
import spacy

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Функция для анализа структуры предложения
def analyze_sentence_structure(sentence):
    # Извлечение лингвистических характеристик из предложения
    tokens = [token.text for token in sentence]  # Токены
    pos_tags = [token.pos_ for token in sentence]  # Часть речи (POS-теги)
    dependencies = [
        (token.text, token.dep_, token.head.text) for token in sentence
    ]  # Зависимости
    return tokens, pos_tags, dependencies


# Функция для обработки текстов
def process_texts(texts):
    # Загрузка русской языковой модели Spacy
    nlp = spacy.load("ru_core_news_sm")

    total_sentence_count = 0  # Общее количество предложений
    tokens_count = Counter()  # Счетчик токенов
    pos_tags_count = Counter()  # Счетчик POS-тегов
    dependencies_count = Counter()  # Счетчик зависимостей

    # Итерация по каждому тексту в списке
    for text_content in texts:
        # Использование Spacy для сегментации и анализа предложений
        doc = nlp(text_content)
        for sentence in doc.sents:
            tokens, pos_tags, dependencies = analyze_sentence_structure(sentence)
            tokens_count.update(tokens)  # Обновление счетчика токенов
            pos_tags_count.update(pos_tags)  # Обновление счетчика POS-тегов
            dependencies_count.update(dependencies)  # Обновление счетчика зависимостей
            total_sentence_count += 1  # Увеличение общего количества предложений

    return {
        "sentence_count": total_sentence_count,  # Возвращение общего количества предложений
        "tokens": tokens_count,  # Возвращение счетчика токенов
        "pos_tags": pos_tags_count,  # Возвращение счетчика POS-тегов
        "dependencies": dependencies_count,  # Возвращение счетчика зависимостей
    }


# Функция для параллельной обработки текстов
def parallel_process_texts(column_data, num_processes=None):
    if num_processes is None:
        num_processes = 16  # Установка количества процессов по умолчанию
    chunk_size = (
        len(column_data) // num_processes
    )  # Определение размера части для каждого процесса
    chunks = [
        column_data[i : i + chunk_size] for i in range(0, len(column_data), chunk_size)
    ]  # Разделение данных на части
    with Pool(processes=num_processes) as pool:
        results = pool.map(
            process_texts, chunks
        )  # Параллельная обработка частей данных
    # Объединение результатов всех процессов
    combined_results = {
        "sentence_count": sum(
            r["sentence_count"] for r in results
        ),  # Объединение общего количества предложений
        "tokens": sum(
            (r["tokens"] for r in results), Counter()
        ),  # Объединение счетчика токенов
        "pos_tags": sum(
            (r["pos_tags"] for r in results), Counter()
        ),  # Объединение счетчика POS-тегов
        "dependencies": sum(
            (r["dependencies"] for r in results), Counter()
        ),  # Объединение счетчика зависимостей
    }
    return combined_results  # Возвращение объединенных результатов


def main(df):
    # Параллельная обработка столбцов 'text' и 'text_prep'
    logging.info(
        "Processing 'text' column"
    )  # Логирование начала обработки столбца 'text'
    results_text = parallel_process_texts(df["text"])

    logging.info(
        "Processing 'text_prep' column"
    )  # Логирование начала обработки столбца 'text_prep'
    results_text_prep = parallel_process_texts(df["text_prep"])

    # Логирование и печать результатов
    logging.info("Total count of sentences (text): %d", results_text["sentence_count"])
    logging.info("Total count of unique tokens (text): %d", len(results_text["tokens"]))
    logging.info(
        "Total count of unique POS tags (text): %d", len(results_text["pos_tags"])
    )
    logging.info(
        "Total count of unique dependency relations (text): %d",
        len(results_text["dependencies"]),
    )

    logging.info(
        "Total count of sentences (text_prep): %d", results_text_prep["sentence_count"]
    )
    logging.info(
        "Total count of unique tokens (text_prep): %d", len(results_text_prep["tokens"])
    )
    logging.info(
        "Total count of unique POS tags (text_prep): %d",
        len(results_text_prep["pos_tags"]),
    )
    logging.info(
        "Total count of unique dependency relations (text_prep): %d",
        len(results_text_prep["dependencies"]),
    )

    # Показ топ-5 POS-тегов
    logging.info("Top 5 POS tags (text): %s", results_text["pos_tags"].most_common(5))
    logging.info(
        "Top 5 POS tags (text_prep): %s", results_text_prep["pos_tags"].most_common(5)
    )


if __name__ == "__main__":
    # Загрузка данных из CSV файла
    df = pd.read_csv("extracted_cases_preprocessed.csv")
    main(df)
