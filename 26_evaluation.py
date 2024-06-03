import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Загрузка датасетов
auto_labeled = pd.read_excel(
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\sample_dataset.xlsx"
)
gold_standard = pd.read_excel(
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\таблица_проверочная.xlsx"
)

auto_labeled = auto_labeled.sort_values(by="ID").reset_index(drop=True)
gold_standard = gold_standard.sort_values(by="ID").reset_index(drop=True)

# Проверка совпадения приговоров
id_match = auto_labeled["ID"].equals(gold_standard["ID"])
print(id_match)


columns_to_compare = [
    "is_homicide",
    "many_murderers",
    "cr_sex",
    "vi_sex",
    "cr_other_people_around",
    "cr_previous_conviction",
    "cr_getaway",
]

# Объединение для синхронного удаления строк с NaN
combined = pd.concat(
    [gold_standard[columns_to_compare], auto_labeled[columns_to_compare]],
    axis=1,
    keys=["gold", "auto"],
)
combined = combined.dropna()

y_true = combined["gold"]
y_pred = combined["auto"]

results = {}

# Для хранения метрик в Excel
metrics_list = []

for column in columns_to_compare:
    y_true_col = y_true[column]
    y_pred_col = y_pred[column]

    accuracy = accuracy_score(y_true_col, y_pred_col)
    precision = precision_score(
        y_true_col, y_pred_col, average="weighted", zero_division=1
    )
    recall = recall_score(y_true_col, y_pred_col, average="weighted", zero_division=1)
    f1 = f1_score(y_true_col, y_pred_col, average="weighted", zero_division=1)

    conf_matrix = confusion_matrix(y_true_col, y_pred_col)

    results[column] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": classification_report(
            y_true_col, y_pred_col, zero_division=1
        ),
    }

    metrics_list.append(
        {
            "Metric": column,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    )
# Вывод результатов
for column, metrics in results.items():
    print(f"Metrics for {column}:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("Classification Report:")
    print(metrics["classification_report"])
    print("\n")

# Создание датафрейма из списка метрик
metrics_df = pd.DataFrame(metrics_list)

# Запись в Excel
metrics_df.to_excel("metrics_results.xlsx", index=False)
