import statistics  # Импортируем модуль для вычисления статистических параметров

import pandas as pd  # Импортируем модуль для работы с данными в формате DataFrame
from nltk.tokenize import (
    word_tokenize,  # Импортируем функцию для токенизации текста на слова из библиотеки NLTK
)

# Загружаем данные из CSV файла в DataFrame
df = pd.read_csv("extracted_cases_preprocessed.csv")


# Функция для подсчета количества слов в тексте
def count_words(text):
    words = word_tokenize(text)  # Разбиваем текст на слова
    return len(words)  # Возвращаем количество слов


# Вычисляем количество слов для колонок 'text' и 'text_prep'
df["word_count_text"] = df["text"].apply(count_words)
df["word_count_text_prep"] = df["text_prep"].apply(count_words)

# Вычисляем общую сумму, среднее и медианное количество слов для каждой колонки
total_words_text = df["word_count_text"].sum()
mean_words_text = statistics.mean(df["word_count_text"])
median_words_text = statistics.median(df["word_count_text"])

total_words_text_prep = df["word_count_text_prep"].sum()
mean_words_text_prep = statistics.mean(df["word_count_text_prep"])
median_words_text_prep = statistics.median(df["word_count_text_prep"])

# Выводим результаты для обеих колонок
print("Total words in 'text':", total_words_text)
print("Mean words per 'text':", mean_words_text)
print("Median words per 'text':", median_words_text)

print("Total words in 'text_prep':", total_words_text_prep)
print("Mean words per 'text_prep':", mean_words_text_prep)
print("Median words per 'text_prep':", median_words_text_prep)
