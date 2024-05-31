import statistics  # Импортируем модуль для вычисления статистических параметров

import pandas as pd  # Импортируем модуль для работы с данными в формате DataFrame

# Загружаем данные из CSV файла в DataFrame
df = pd.read_csv("extracted_cases_preprocessed.csv")

# Извлекаем колонку 'text' и 'text_prep' из DataFrame
texts = df["text"]
texts_prep = df["text_prep"]

# Рассчитываем количество символов в каждом тексте
character_counts = texts.apply(len)
character_counts_prep = texts_prep.apply(len)

# Вычисляем общую сумму, среднее и медианное количество символов в текстах
total_characters = character_counts.sum()
mean_characters = statistics.mean(character_counts)
median_characters = statistics.median(character_counts)

total_characters_prep = character_counts_prep.sum()
mean_characters_prep = statistics.mean(character_counts_prep)
median_characters_prep = statistics.median(character_counts_prep)

# Выводим результаты на экран
print("Total characters in all texts:", total_characters)
print("Mean characters per text:", mean_characters)
print("Median characters per text:", median_characters)

print("Total characters in all texts:", total_characters_prep)
print("Mean characters per text:", mean_characters_prep)
print("Median characters per text:", median_characters_prep)
