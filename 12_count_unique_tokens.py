import pandas as pd  # Импортируем модуль для работы с данными в формате DataFrame
from nltk.tokenize import (
    word_tokenize,  # Импортируем функцию для токенизации текста на слова из библиотеки NLTK
)

# Загружаем данные из CSV файла в DataFrame
df = pd.read_csv("extracted_cases_preprocessed.csv")


# Функция для токенизации текста и возврата уникальных токенов
def unique_tokens_in_text(text):
    tokens = word_tokenize(text)  # Разбиваем текст на токены (слова)
    return set(tokens)  # Возвращаем множество уникальных токенов


# Применяем функцию к колонке 'text' и накапливаем уникальные токены
unique_tokens_text = set()  # Инициализируем пустое множество для уникальных токенов
df["text"].apply(
    lambda x: unique_tokens_text.update(unique_tokens_in_text(x))
)  # Применяем функцию к каждой строке колонки 'text' и обновляем множество

# Применяем функцию к колонке 'text_prep' и накапливаем уникальные токены
unique_tokens_text_prep = (
    set()
)  # Инициализируем пустое множество для уникальных токенов
df["text_prep"].apply(
    lambda x: unique_tokens_text_prep.update(unique_tokens_in_text(x))
)  # Применяем функцию к каждой строке колонки 'text_prep' и обновляем множество

# Подсчитываем количество уникальных токенов для каждой колонки
num_unique_tokens_text = len(
    unique_tokens_text
)  # Количество уникальных токенов в колонке 'text'
num_unique_tokens_text_prep = len(
    unique_tokens_text_prep
)  # Количество уникальных токенов в колонке 'text_prep'

# Выводим результаты для обеих колонок
print("Number of unique tokens in 'text':", num_unique_tokens_text)
print("Number of unique tokens in 'text_prep':", num_unique_tokens_text_prep)
