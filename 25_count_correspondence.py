import pandas as pd

# Load the datasets
sample_dataset = pd.read_excel(
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\sample_dataset.xlsx"
)
verification_dataset = pd.read_excel(
    r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\таблица_проверочная.xlsx"
)

sample_dataset = sample_dataset.sort_values(by="ID").reset_index(drop=True)
verification_dataset = verification_dataset.sort_values(by="ID").reset_index(drop=True)

# Check if all names in ID columns match
id_match = sample_dataset["ID"].equals(verification_dataset["ID"])
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
        sample_dataset[column] == verification_dataset[column]
    ).mean() * 100

# Prepare results
results = {"ID Match": id_match, "Matching Percentages": matching_percentage}

print(results)
