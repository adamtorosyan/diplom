import os  # Импортируем os для работы с файловой системой
from collections import Counter  # Импортируем Counter для подсчета токенов
from multiprocessing import Pool  # Импортируем Pool для параллельной обработки

import matplotlib.pyplot as plt  # Импортируем matplotlib для построения графиков
import nltk  # Импортируем nltk для обработки естественного языка
import pandas as pd  # Импортируем pandas для работы с данными
from nltk.tokenize import (
    word_tokenize,  # Импортируем word_tokenize для токенизации текста
)

# Загрузка ресурса 'punkt' для токенизации
nltk.download("punkt")


def plot_token_frequencies(tokens, frequencies, title, filename):
    # Функция для построения графика частотности токенов
    plt.figure(figsize=(10, 6))
    plt.bar(tokens, frequencies)  # Строим столбчатую диаграмму
    plt.xlabel("Tokens")  # Устанавливаем подпись оси X
    plt.ylabel("Frequency")  # Устанавливаем подпись оси Y
    plt.yscale("log")  # Устанавливаем логарифмическую шкалу для оси Y
    plt.title(title)  # Устанавливаем заголовок графика
    plt.xticks(
        rotation=45, ha="right"
    )  # Поворачиваем метки на оси X для лучшей читаемости
    plt.tight_layout()  # Оптимизируем размещение элементов на графике
    plt.savefig(
        filename, dpi=300
    )  # Сохраняем график как изображение с высоким разрешением
    plt.close()  # Закрываем график


def count_tokens(texts):
    # Функция для подсчета токенов в текстах
    token_counter = Counter()  # Инициализируем Counter
    for text in texts:
        tokens = word_tokenize(text)  # Токенизируем текст
        token_counter.update(tokens)  # Обновляем счетчик токенов
    return token_counter  # Возвращаем счетчик токенов


def process_chunk(texts):
    # Функция для обработки чанка текстов
    counter = Counter()  # Инициализируем Counter
    for text in texts:
        counter.update(word_tokenize(text))  # Токенизируем текст и обновляем счетчик
    return counter  # Возвращаем счетчик токенов для чанка


def parallel_process_texts(texts, num_processes=None):
    # Функция для параллельной обработки текстов
    if num_processes is None:
        num_processes = 15  # Устанавливаем количество процессов по умолчанию
    chunk_size = len(texts) // num_processes  # Определяем размер чанка
    chunks = [
        texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)
    ]  # Разбиваем тексты на чанки
    with Pool(processes=num_processes) as pool:
        counters = pool.map(process_chunk, chunks)  # Параллельно обрабатываем чанки
    total_counter = Counter()  # Инициализируем общий Counter
    for counter in counters:
        total_counter.update(counter)  # Объединяем результаты из всех процессов
    return total_counter  # Возвращаем общий счетчик токенов


def main(df):
    # Основная функция
    # Обработка колонки 'text' в параллельном режиме
    token_counter_text = parallel_process_texts(df["text"])

    # Обработка колонки 'text_prep' в параллельном режиме
    token_counter_text_prep = parallel_process_texts(df["text_prep"])

    # Получаем самые частотные токены и их частоты для колонки 'text'
    num_tokens_to_visualize = 20  # Количество токенов для визуализации
    most_common_tokens_text = token_counter_text.most_common(num_tokens_to_visualize)
    tokens_text, frequencies_text = zip(*most_common_tokens_text)

    # Строим график для колонки 'text'
    plot_token_frequencies(
        tokens_text,
        frequencies_text,
        "Top 20 Most Frequent Words in 'text'",
        "top_20_most_common_words_text.png",
    )

    # Получаем самые частотные токены и их частоты для колонки 'text_prep'
    most_common_tokens_text_prep = token_counter_text_prep.most_common(
        num_tokens_to_visualize
    )
    tokens_text_prep, frequencies_text_prep = zip(*most_common_tokens_text_prep)

    # Строим график для колонки 'text_prep'
    plot_token_frequencies(
        tokens_text_prep,
        frequencies_text_prep,
        "Top 20 Most Frequent Words in 'text_prep'",
        "top_20_most_common_words_text_prep.png",
    )

    # Указываем путь для сохранения изображений
    download_directory = "visualizations"
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)  # Создаем директорию, если она не существует

    # Перемещаем сгенерированные изображения в директорию
    os.replace(
        "top_20_most_common_words_text.png",
        os.path.join(download_directory, "top_20_most_common_words_text.png"),
    )
    os.replace(
        "top_20_most_common_words_text_prep.png",
        os.path.join(download_directory, "top_20_most_common_words_text_prep.png"),
    )


if __name__ == "__main__":
    df = pd.read_csv(
        "extracted_cases_preprocessed.csv"
    )  # Загружаем данные из CSV файла
    main(df)  # Вызываем основную функцию
