import os
from collections import Counter
from multiprocessing import Pool

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

# Загрузка ресурса 'punkt' для токенизации
nltk.download("punkt")


# Функция для построения и сохранения графика частотности токенов
def plot_token_frequencies(tokens, frequencies, title, filename):
    plt.figure(figsize=(10, 6))  # Задание размера фигуры
    plt.bar(tokens, frequencies)  # Построение столбчатой диаграммы
    plt.xlabel("Tokens")  # Метка оси X
    plt.ylabel("Frequency")  # Метка оси Y
    plt.title(title)  # Заголовок графика
    plt.xticks(rotation=45, ha="right")  # Поворот меток оси X для лучшей читаемости
    plt.tight_layout()  # Автоматическая настройка параметров графика для наилучшего размещения
    plt.savefig(
        filename, dpi=300
    )  # Сохранение графика как изображения с разрешением 300 dpi
    plt.close()  # Закрытие текущей фигуры


# Функция для подсчета частотности токенов в списке текстов
def count_tokens(texts):
    token_counter = Counter()  # Создание счетчика для токенов
    for text in texts:
        tokens = word_tokenize(text)  # Токенизация текста
        token_counter.update(tokens)  # Обновление счетчика токенами из текста
    return token_counter  # Возвращение счетчика токенов


# Функция для обработки части текстов и подсчета токенов
def process_chunk(texts):
    counter = Counter()  # Создание счетчика для токенов
    for text in texts:
        counter.update(word_tokenize(text))  # Обновление счетчика токенами из текста
    return counter  # Возвращение счетчика токенов


# Функция для параллельной обработки текстов с использованием нескольких процессов
def parallel_process_texts(texts, num_processes=None):
    if num_processes is None:
        num_processes = 15  # Установка количества процессов по умолчанию
    chunk_size = len(texts) // num_processes  # Размер части для каждого процесса
    chunks = [
        texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)
    ]  # Разделение текстов на части
    with Pool(processes=num_processes) as pool:
        counters = pool.map(
            process_chunk, chunks
        )  # Параллельная обработка частей текстов
    total_counter = Counter()  # Создание общего счетчика для всех токенов
    for counter in counters:
        total_counter.update(
            counter
        )  # Обновление общего счетчика токенами из всех частей
    return total_counter  # Возвращение общего счетчика токенов


def main(df):
    # Параллельная обработка текстов из столбца 'text'
    token_counter_text = parallel_process_texts(df["text"])

    # Параллельная обработка текстов из столбца 'text_prep'
    token_counter_text_prep = parallel_process_texts(df["text_prep"])

    # Получение самых частых токенов и их частот для столбца 'text'
    num_tokens_to_visualize = 20  # Количество токенов для визуализации (можно изменить)
    most_common_tokens_text = token_counter_text.most_common(num_tokens_to_visualize)
    tokens_text, frequencies_text = zip(*most_common_tokens_text)

    # Построение графика для столбца 'text'
    plot_token_frequencies(
        tokens_text,
        frequencies_text,
        "Top 20 Most Frequent Words in 'text'",  # Заголовок графика
        "top_20_most_common_words_text.png",  # Имя файла для сохранения
    )

    # Получение самых частых токенов и их частот для столбца 'text_prep'
    most_common_tokens_text_prep = token_counter_text_prep.most_common(
        num_tokens_to_visualize
    )
    tokens_text_prep, frequencies_text_prep = zip(*most_common_tokens_text_prep)

    # Построение графика для столбца 'text_prep'
    plot_token_frequencies(
        tokens_text_prep,
        frequencies_text_prep,
        "Top 20 Most Frequent Words in 'text_prep'",  # Заголовок графика
        "top_20_most_common_words_text_prep.png",  # Имя файла для сохранения
    )

    # Создание директории для сохранения изображений, если она не существует
    download_directory = "visualizations"
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    # Перемещение сгенерированных изображений в директорию для скачивания
    os.replace(
        "top_20_most_common_words_text.png",
        os.path.join(download_directory, "top_20_most_common_words_text.png"),
    )
    os.replace(
        "top_20_most_common_words_text_prep.png",
        os.path.join(download_directory, "top_20_most_common_words_text_prep.png"),
    )


if __name__ == "__main__":
    # Загрузка данных из CSV файла
    df = pd.read_csv("extracted_cases_preprocessed.csv")
    main(df)
