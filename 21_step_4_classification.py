import joblib
import optuna
import pandas as pd
from imblearn.over_sampling import ADASYN
from optuna.integration import OptunaSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from xgboost import XGBClassifier


# Функция для подготовки данных
def prepare_data(df, target_col):
    """
    Подготовка данных для обучения и валидации.

    Аргументы:
    df -- DataFrame с данными
    target_col -- Целевая колонка для классификации

    Возвращает:
    X_train, X_val, y_train, y_val -- Разделенные на обучающие и валидационные наборы данных
    tfidf_vectorizer -- TF-IDF векторизатор
    """
    X = df["text_prep"]
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    return X_train_tfidf, X_val_tfidf, y_train, y_val, tfidf_vectorizer


# Функция для обучения и оценки моделей
def train_and_evaluate_models(X_train, X_val, y_train, y_val, tfidf_vectorizer, target):
    """
    Обучение и оценка моделей Random Forest и XGBoost.

    Аргументы:
    X_train -- Обучающие данные
    X_val -- Валидационные данные
    y_train -- Метки обучающих данных
    y_val -- Метки валидационных данных
    tfidf_vectorizer -- TF-IDF векторизатор
    target -- Целевая переменная

    Возвращает:
    results -- Результаты в виде списка словарей
    """
    results = []

    # Обучение модели Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Предсказания и оценка
    y_pred_rf = rf_model.predict(X_val)
    evaluate_model(y_val, y_pred_rf, "Random Forest", target, results)

    # Сохранение модели и векторизатора
    joblib.dump(rf_model, f"rf_model_{target}.pkl")
    joblib.dump(tfidf_vectorizer, f"tfidf_vectorizer_{target}.pkl")

    # Обучение модели XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=10, learning_rate=0.1, random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Предсказания и оценка
    y_pred_xgb = xgb_model.predict(X_val)
    evaluate_model(y_val, y_pred_xgb, "XGBoost", target, results)

    # Сохранение модели
    joblib.dump(xgb_model, f"xgb_model_{target}.pkl")

    return results


# Функция для оценки модели
def evaluate_model(y_true, y_pred, model_name, target, results):
    """
    Оценка модели и добавление результатов в список.

    Аргументы:
    y_true -- Истинные метки
    y_pred -- Предсказанные метки
    model_name -- Название модели
    target -- Целевая переменная
    results -- Список для сохранения результатов
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"{model_name} - {target}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    results.append(
        {
            "target": target,
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": classification_report(y_true, y_pred),
        }
    )


# Функция для обучения модели RuBERT с гиперпараметрами
def train_rubert_model(df, target_col):
    """
    Обучение модели RuBERT с использованием гиперпараметров.

    Аргументы:
    df -- DataFrame с данными
    target_col -- Целевая колонка для классификации

    Возвращает:
    results -- Результаты в виде списка словарей
    """
    results = []

    # Подготовка данных
    X = df["text_prep"].tolist()
    y = df[target_col].tolist()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=512)

    train_dataset = CustomDataset(train_encodings, y_train)
    val_dataset = CustomDataset(val_encodings, y_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        "DeepPavlov/rubert-base-cased", num_labels=3
    )

    # Настройка параметров обучения
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Обучение модели
    trainer.train()

    # Оценка модели
    eval_results = trainer.evaluate()
    print(f"Evaluation Results for {target_col}:")
    print(eval_results)

    # Генерация отчета классификации
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    report = classification_report(
        y_val,
        preds,
        labels=[0, 1, 2],
        target_names=["class_0", "class_1", "class_2"],
        zero_division=0,
    )
    print(f"Classification Report for {target_col}:\n{report}")

    results.append(
        {
            "target": target_col,
            "model": "RuBERT",
            "accuracy": eval_results["eval_accuracy"],
            "precision": eval_results["eval_precision"],
            "recall": eval_results["eval_recall"],
            "f1": eval_results["eval_f1"],
            "classification_report": report,
        }
    )

    model.save_pretrained(f"rubert_model_{target_col}")
    tokenizer.save_pretrained(f"rubert_tokenizer_{target_col}")

    return results


# Пользовательский датасет для RuBERT
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


# Функция для вычисления метрик
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="weighted"
    )
    accuracy = accuracy_score(labels, pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Основной код
if __name__ == "__main__":
    df = pd.read_csv("data.csv")

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
        X_train, X_val, y_train, y_val, tfidf_vectorizer = prepare_data(df, target)
        results = train_and_evaluate_models(
            X_train, X_val, y_train, y_val, tfidf_vectorizer, target
        )
        all_results.extend(results)

    # Обучение модели RuBERT
    rubert_results = train_rubert_model(df, "is_homicide")
    all_results.extend(rubert_results)

    # Сохранение всех результатов в DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("model_results.csv", index=False)
