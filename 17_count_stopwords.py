import logging
import os
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Настройка логирования
logging.basicConfig(
    filename="stopwords_count_df.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Загрузка стоп-слов NLTK для русского языка
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))


# Функция для обработки части текстов
def process_text_chunk(chunk):
    stop_words_count = 0  # Общее количество стоп-слов
    stop_words_freq = {}  # Частотность стоп-слов

    # Итерация по текстам в части
    for text in chunk:
        # Токенизация текста на слова
        words = word_tokenize(text, language="russian")

        # Подсчет стоп-слов
        stop_words_count += len([word for word in words if word.lower() in stop_words])

        # Подсчет вхождений каждого стоп-слова
        for word in words:
            if word.lower() in stop_words:
                stop_words_freq[word.lower()] = stop_words_freq.get(word.lower(), 0) + 1

    return (
        stop_words_count,
        stop_words_freq,
    )  # Возвращение количества и частотности стоп-слов


# Функция для параллельной обработки текстов
def parallel_process_texts(column_data, num_processes=None):
    if num_processes is None:
        num_processes = 15  # Установка количества процессов по умолчанию
    chunk_size = len(column_data) // num_processes  # Размер части для каждого процесса
    chunks = [
        column_data[i : i + chunk_size] for i in range(0, len(column_data), chunk_size)
    ]  # Разделение данных на части
    with Pool(processes=num_processes) as pool:
        results = pool.map(
            process_text_chunk, chunks
        )  # Параллельная обработка частей данных
    # Объединение результатов всех процессов
    total_stop_words_count = sum(r[0] for r in results)  # Общее количество стоп-слов
    total_stop_words_freq = {}  # Объединенная частотность стоп-слов
    for _, freq_dict in results:
        for word, count in freq_dict.items():
            total_stop_words_freq[word] = total_stop_words_freq.get(word, 0) + count
    return (
        total_stop_words_count,
        total_stop_words_freq,
    )  # Возвращение общего количества и частотности стоп-слов


# Главная функция
def main(df):
    # Параллельная обработка столбцов 'text' и 'text_prep'
    logging.info(
        "Processing 'text' column"
    )  # Логирование начала обработки столбца 'text'
    stop_words_count_text, stop_words_freq_text = parallel_process_texts(df["text"])

    logging.info(
        "Processing 'text_prep' column"
    )  # Логирование начала обработки столбца 'text_prep'
    stop_words_count_text_prep, stop_words_freq_text_prep = parallel_process_texts(
        df["text_prep"]
    )

    # Логирование общего количества стоп-слов для каждого столбца
    logging.info(f"Total stop words count for 'text': {stop_words_count_text}")
    logging.info(
        f"Total stop words count for 'text_prep': {stop_words_count_text_prep}"
    )
    print("Total stop words count for 'text':", stop_words_count_text)
    print("Total stop words count for 'text_prep':", stop_words_count_text_prep)

    # Сортировка и построение графика для столбца 'text' в качестве примера
    sorted_stop_words_freq_text = sorted(
        stop_words_freq_text.items(), key=lambda x: x[1], reverse=True
    )
    top_20_stop_words_text = sorted_stop_words_freq_text[:20]  # Топ-20 стоп-слов
    stop_words_text, frequencies_text = zip(*top_20_stop_words_text)

    plt.figure(figsize=(10, 6))
    plt.barh(
        stop_words_text, frequencies_text, color="skyblue"
    )  # Построение горизонтальной столбчатой диаграммы
    plt.xlabel("Frequency")
    plt.ylabel("Stop Words")
    plt.title("Top 20 Stop Words in 'text' column")
    plt.gca().invert_yaxis()  # Инвертирование оси Y

    plt.savefig("top_20_stop_words_text.png", bbox_inches="tight")  # Сохранение графика
    plt.savefig(
        "visualizations/top_20_stop_words_text.png", bbox_inches="tight"
    )  # Сохранение в директории visualizations
    plt.show()

    # Аналогично, построение графика для столбца 'text_prep'
    sorted_stop_words_freq_text_prep = sorted(
        stop_words_freq_text_prep.items(), key=lambda x: x[1], reverse=True
    )
    top_20_stop_words_text_prep = sorted_stop_words_freq_text_prep[
        :20
    ]  # Топ-20 стоп-слов
    stop_words_text_prep, frequencies_text_prep = zip(*top_20_stop_words_text_prep)

    plt.figure(figsize=(10, 6))
    plt.barh(
        stop_words_text_prep, frequencies_text_prep, color="skyblue"
    )  # Построение горизонтальной столбчатой диаграммы
    plt.xlabel("Frequency")
    plt.ylabel("Stop Words")
    plt.title("Top 20 Stop Words in 'text_prep' column")
    plt.gca().invert_yaxis()  # Инвертирование оси Y

    plt.savefig(
        "top_20_stop_words_text_prep.png", bbox_inches="tight"
    )  # Сохранение графика
    plt.savefig(
        "visualizations/top_20_stop_words_text_prep.png", bbox_inches="tight"
    )  # Сохранение в директории visualizations
    plt.show()


# Запуск главной функции
if __name__ == "__main__":
    df = pd.read_csv("extracted_cases_preprocessed.csv")  # Загрузка данных из CSV файла
    main(df)
