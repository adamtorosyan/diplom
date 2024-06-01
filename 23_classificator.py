# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1S-A4WHcDUdNeMdvLASYS10XRAQ23k1v_
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
import joblib
from google.colab import drive

drive.mount('/content/drive')

base_path = '/content/drive/My Drive/diplom/'

# Загрузка данных
df1 = pd.read_csv(base_path + 'df1.csv')  # Укажите правильный путь
unmarked_data = pd.read_csv(base_path + 'extracted_cases_preprocessed.csv')  # Укажите правильный путь

# Настройка данных
X = df1['text_prep']
y = df1['is_homicide']

# Настройка данных
X = df1['text_prep']
y = df1['is_homicide']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Подготовка данных для tf-idf
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Применение SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

# Пайплайн для XGBoost
pipeline1 = Pipeline([
    ('xgb', XGBClassifier(colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.8, random_state=42))
])

# Обучение модели
pipeline1.fit(X_train_res, y_train_res)

# Предсказание и оценка качества модели
y_pred = pipeline1.predict(X_test_tfidf)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Сохранение модели
joblib.dump(pipeline1, 'pipeline1_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer_1.pkl')

# Загрузка данных
df2 = pd.read_csv(base_path + 'df2.csv')  # Укажите правильный путь

# Настройка данных
X = df2['text_prep']
y = df2['many_murderers']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Подготовка данных для tf-idf
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Применение SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

# Пайплайн для XGBoost
pipeline2 = Pipeline([
    ('xgb', XGBClassifier(colsample_bytree=0.8, learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.8, random_state=42))
])

# Обучение модели
pipeline2.fit(X_train_res, y_train_res)

# Предсказание и оценка качества модели
y_pred = pipeline2.predict(X_test_tfidf)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Сохранение модели
joblib.dump(pipeline2, 'pipeline2_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer_2.pkl')

import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy as np
import joblib

# Загрузка данных
df3 = pd.read_csv(base_path + 'df3.csv')  # Укажите правильный путь

# Применение feature engineering
def feature_engineering(df):
    df['text_length'] = df['text_prep'].apply(len)
    df['word_count'] = df['text_prep'].apply(lambda x: len(x.split()))
    df['sentiment'] = df['text_prep'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

df3 = feature_engineering(df3)

# Подготовка данных для tf-idf
tfidf = TfidfVectorizer(max_features=5000)

# Настройка гиперпараметров для моделей
param_grid_xgb = {
    'xgb__colsample_bytree': [0.8, 0.9],
    'xgb__learning_rate': [0.1, 0.2],
    'xgb__max_depth': [3, 4],
    'xgb__n_estimators': [100, 200, 300],
    'xgb__subsample': [0.8, 0.9]
}

param_grid_rf = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

targets = ['cr_sex', 'vi_sex', 'cr_other_people_around', 'cr_previous_conviction', 'cr_getaway']
models = {}

for target in targets:
    X = df3['text_prep']
    y = df3[target]

    # Применение tf-idf
    X_tfidf = tfidf.fit_transform(X)
    X_additional_features = df3[['text_length', 'word_count', 'sentiment']].values
    X_combined = np.hstack((X_tfidf.toarray(), X_additional_features))

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Определение модели и гиперпараметров в зависимости от целевой колонки
    if target in ['cr_sex', 'vi_sex']:
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('xgb', XGBClassifier(random_state=42))
        ])
        param_grid = param_grid_xgb
    else:
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42))
        ])
        param_grid = param_grid_rf

    # Настройка гиперпараметров
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Лучшая модель и её гиперпараметры
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f'Best params for {target}: {best_params}')

    # Оценка качества модели
    y_pred = best_model.predict(X_test)
    print(f'Accuracy for {target}: {accuracy_score(y_test, y_pred)}')

    # Сохранение модели
    models[target] = best_model
    joblib.dump(best_model, f'{target}_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer_3.pkl')

# Загрузка моделей и векторизаторов
pipeline1 = joblib.load('pipeline1_model.pkl')
pipeline2 = joblib.load('pipeline2_model.pkl')
tfidf_vectorizer_1 = joblib.load('tfidf_vectorizer_1.pkl')
tfidf_vectorizer_2 = joblib.load('tfidf_vectorizer_2.pkl')
tfidf_vectorizer_3 = joblib.load('tfidf_vectorizer_3.pkl')

# Загрузка моделей для остальных целевых колонок
targets = ['cr_sex', 'vi_sex', 'cr_other_people_around', 'cr_previous_conviction', 'cr_getaway']
models = {}
for target in targets:
    models[target] = joblib.load(f'{target}_model.pkl')

# Применение feature engineering для целевых колонок из targets
def feature_engineering(df):
    df['text_length'] = df['text_prep'].apply(len)
    df['word_count'] = df['text_prep'].apply(lambda x: len(x.split()))
    df['sentiment'] = df['text_prep'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

unmarked_data = feature_engineering(unmarked_data)

# Применение моделей для разметки данных
def apply_models(row):
    predictions = {}
    row_text_prep = row['text_prep']

    # Преобразование текста с помощью tfidf для pipeline1
    row_text_tfidf_1 = tfidf_vectorizer_1.transform([row_text_prep])

    # Предсказание is_homicide
    is_homicide_pred = pipeline1.predict(row_text_tfidf_1)[0]
    if is_homicide_pred == 0:
        predictions['is_homicide'] = 0
        return predictions

    predictions['is_homicide'] = 1

    # Преобразование текста с помощью tfidf для pipeline2
    row_text_tfidf_2 = tfidf_vectorizer_2.transform([row_text_prep])

    # Предсказание many_murderers
    many_murderers_pred = pipeline2.predict(row_text_tfidf_2)[0]
    if many_murderers_pred == 1:
        predictions['many_murderers'] = 1
        return predictions

    predictions['many_murderers'] = 0

    # Для остальных колонок, включая дополнительные признаки
    row_text_tfidf_3 = tfidf_vectorizer_3.transform([row_text_prep])
    row_additional_features = np.array([[row['text_length'], row['word_count'], row['sentiment']]])
    row_combined = np.hstack((row_text_tfidf_3.toarray(), row_additional_features))

    for target in targets:
        predictions[target] = models[target].predict(row_combined)[0]

    return predictions

# Применение ко всем данным
unmarked_data['predictions'] = unmarked_data.apply(apply_models, axis=1)

# Разметка данных
for target in ['is_homicide', 'many_murderers'] + targets:
    unmarked_data[target] = unmarked_data['predictions'].apply(lambda x: x.get(target, None))

unmarked_data.drop(columns=['predictions'], inplace=True)

# Сохранение размеченных данных
unmarked_data.to_csv(base_path + 'labeled_unmarked_data.csv', index=False)