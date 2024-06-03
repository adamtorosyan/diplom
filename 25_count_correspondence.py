import pandas as pd

# Load the datasets
auto_labeled = pd.read_excel(
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\sample_dataset.xlsx"
)
gold_standard = pd.read_excel(
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\таблица_проверочная.xlsx"
)

auto_labeled = auto_labeled.sort_values(by="ID").reset_index(drop=True)
gold_standard = gold_standard.sort_values(by="ID").reset_index(drop=True)

# Check if all names in ID columns match
id_match = auto_labeled["ID"].equals(gold_standard["ID"])
print(id_match)
# Calculate the percentage of matching columns
matching_columns = [
    "is_homicide",
    "many_murderers",
    "cr_sex",
    "vi_sex",
    "cr_other_people_around",
    "cr_previous_conviction",
    "cr_getaway",
]
matching_percentage = {}
for column in matching_columns:
    matching_percentage[column] = (
        auto_labeled[column] == gold_standard[column]
    ).mean() * 100

# Prepare results
results = {"ID Match": id_match, "Matching Percentages": matching_percentage}

print(results)
