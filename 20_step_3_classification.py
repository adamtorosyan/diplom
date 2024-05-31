import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    BertModel,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# Функция для извлечения BERT эмбеддингов
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


# Функция для подготовки данных и инженерии признаков
def prepare_data(df):
    """
    Подготовка данных и инженерия признаков.

    Аргументы:
    df -- DataFrame с данными

    Возвращает:
    df -- DataFrame с добавленными признаками
    tfidf_vectorizer -- TF-IDF векторизатор
    """
    df["text_length"] = df["text_prep"].apply(len)
    df["word_count"] = df["text_prep"].apply(lambda x: len(x.split()))

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertModel.from_pretrained("bert-base-multilingual-cased")

    df["bert_embeddings"] = df["text_prep"].apply(
        lambda x: get_bert_embeddings(x, tokenizer, model).flatten()
    )

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_features = tfidf_vectorizer.fit_transform(df["text_prep"])

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                tfidf_features.toarray(),
                columns=tfidf_vectorizer.get_feature_names_out(),
            ),
        ],
        axis=1,
    )

    return df, tfidf_vectorizer


# Функция для обучения и оценки моделей
def train_and_evaluate_models(df, target_col):
    """
    Обучение и оценка моделей Random Forest и XGBoost.

    Аргументы:
    df -- DataFrame с данными
    target_col -- Целевая колонка для классификации

    Возвращает:
    results -- Результаты в виде списка словарей
    """
    results = []

    X = df.drop(columns=[target_col, "text", "text_prep"])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Обучение модели Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_res, y_train_res)
    evaluate_model(rf_model, X_val, y_val, "Random Forest", target_col, results)

    # Сохранение модели и векторизатора
    joblib.dump(rf_model, f"rf_model_{target_col}.pkl")

    # Обучение модели XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=10, learning_rate=0.1, random_state=42
    )
    xgb_model.fit(X_train_res, y_train_res)
    evaluate_model(xgb_model, X_val, y_val, "XGBoost", target_col, results)

    # Сохранение модели
    joblib.dump(xgb_model, f"xgb_model_{target_col}.pkl")

    return results


# Функция для оценки модели
def evaluate_model(model, X_val, y_val, model_name, target_col, results):
    """
    Оценка модели и добавление результатов в список.

    Аргументы:
    model -- Обученная модель
    X_val -- Валидационные данные
    y_val -- Метки валидационных данных
    model_name -- Название модели
    target_col -- Целевая переменная
    results -- Список для сохранения результатов
    """
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="weighted")
    recall = recall_score(y_val, y_pred, average="weighted")
    f1 = f1_score(y_val, y_pred, average="weighted")

    print(f"{model_name} - {target_col}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    print("\nClassification Report:\n", classification_report(y_val, y_pred))

    results.append(
        {
            "target": target_col,
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": classification_report(y_val, y_pred),
        }
    )


# Функция для обучения модели BERT с гиперпараметрами
def train_bert_model(df, target_col):
    """
    Обучение модели BERT с использованием гиперпараметров.

    Аргументы:
    df -- DataFrame с данными
    target_col -- Целевая колонка для классификации

    Возвращает:
    results -- Результаты в виде списка словарей
    """
    results = []

    X = df["text_prep"].tolist()
    y = df[target_col].tolist()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)

    train_dataset = CustomDataset(train_encodings, y_train)
    val_dataset = CustomDataset(val_encodings, y_val)

    model = BertModel.from_pretrained("bert-base-multilingual-cased", num_labels=3)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels").to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    predictions, labels, _ = trainer.predict(val_dataset)
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, average="weighted")
    recall = recall_score(y_val, predictions, average="weighted")
    f1 = f1_score(y_val, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    print("\nClassification Report:\n", classification_report(y_val, predictions))

    results.append(
        {
            "target": target_col,
            "model": "BERT",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": classification_report(y_val, predictions),
        }
    )

    return results


# Пользовательский датасет для BERT
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Основной код
if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    # Подготовка данных и инженерия признаков
    df, tfidf_vectorizer = prepare_data(df)

    # Обучение и оценка моделей для каждого целевого столбца
    target_columns = [
        "is_homicide",
        "many_murderers",
        "cr_sex",
        "vi_sex",
        "cr_alco",
        "vi_alco",
        "cr_reason_clarity",
        "cr_other_people_around",
        "cr_children",
        "cr_previous_conviction",
        "cr_getaway",
        "cr_worker",
        "vi_previous_conviction",
    ]

    all_results = []

    for target in target_columns:
        print(f"Training models for {target}")
        results = train_and_evaluate_models(df, target)
        all_results.extend(results)

    # Обучение модели BERT
    bert_results = train_bert_model(df, "is_homicide")
    all_results.extend(bert_results)

    # Сохранение всех результатов в DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("model_results.csv", index=False)
