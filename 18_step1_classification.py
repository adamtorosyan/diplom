import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from xgboost import XGBClassifier

# Загрузка данных
df1 = pd.read_csv("path_to_df1.csv")

# Подготовка данных
# Отделение признаков (текст) и целевой переменной (is_homicide)
X = df1["text_prep"]
y = df1["is_homicide"]

# Разделение данных на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Определение пользовательского датасета
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Инициализация токенизатора и модели
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = BertForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased", num_labels=2
)

# Создание датасетов
train_dataset = CustomDataset(
    X_train.to_numpy(), y_train.to_numpy(), tokenizer, max_len=128
)
val_dataset = CustomDataset(X_val.to_numpy(), y_val.to_numpy(), tokenizer, max_len=128)

# Определение аргументов для тренировки
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Инициализация тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Тренировка модели
trainer.train()

# Оценка модели
predictions, labels, _ = trainer.predict(val_dataset)
predictions = np.argmax(predictions, axis=1)

# Вывод метрик оценки для BERT
bert_accuracy = accuracy_score(y_val, predictions)
bert_precision = precision_score(y_val, predictions)
bert_recall = recall_score(y_val, predictions)
bert_f1 = f1_score(y_val, predictions)

print(f"BERT Accuracy: {bert_accuracy:.4f}")
print(f"BERT Precision: {bert_precision:.4f}")
print(f"BERT Recall: {bert_recall:.4f}")
print(f"BERT F1 Score: {bert_f1:.4f}")

print("\nBERT Classification Report:\n", classification_report(y_val, predictions))

# Подсчет значений целевой переменной
print(df1["is_homicide"].value_counts())

# RandomForest и XGBoost
# Подготовка данных
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Определение модели RandomForest
rf_model = RandomForestClassifier(random_state=42)

# Определение гиперпараметров для Grid Search
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Grid Search для RandomForest
rf_grid = GridSearchCV(
    estimator=rf_model, param_grid=rf_params, cv=3, n_jobs=-1, verbose=2
)
rf_grid.fit(X_train_vec, y_train)

# Лучшая модель RandomForest
best_rf = rf_grid.best_estimator_

# Оценка RandomForest
rf_predictions = best_rf.predict(X_val_vec)
rf_accuracy = accuracy_score(y_val, rf_predictions)
rf_precision = precision_score(y_val, rf_predictions)
rf_recall = recall_score(y_val, rf_predictions)
rf_f1 = f1_score(y_val, rf_predictions)

print(f"RandomForest Accuracy: {rf_accuracy:.4f}")
print(f"RandomForest Precision: {rf_precision:.4f}")
print(f"RandomForest Recall: {rf_recall:.4f}")
print(f"RandomForest F1 Score: {rf_f1:.4f}")

# Определение модели XGBoost
xgb_model = XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)

# Определение гиперпараметров для Grid Search
xgb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Grid Search для XGBoost
xgb_grid = GridSearchCV(
    estimator=xgb_model, param_grid=xgb_params, cv=3, n_jobs=-1, verbose=2
)
xgb_grid.fit(X_train_vec, y_train)

# Лучшая модель XGBoost
best_xgb = xgb_grid.best_estimator_

# Оценка XGBoost
xgb_predictions = best_xgb.predict(X_val_vec)
xgb_accuracy = accuracy_score(y_val, xgb_predictions)
xgb_precision = precision_score(y_val, xgb_predictions)
xgb_recall = recall_score(y_val, xgb_predictions)
xgb_f1 = f1_score(y_val, xgb_predictions)

print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"XGBoost Precision: {xgb_precision:.4f}")
print(f"XGBoost Recall: {xgb_recall:.4f}")
print(f"XGBoost F1 Score: {xgb_f1:.4f}")
