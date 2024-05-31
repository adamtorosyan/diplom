# загружаем необходимые библиотеки
import pandas as pd
import numpy as np

# загружаем кодбук
df = pd.read_excel(r"C:\Users\WarSa\OneDrive\Рабочий стол\диплом\rospravosudie_sou\codebook_27jan2023_ak.xlsx")

# меняем имя ID, это нужно для более простого считыванием файлов из папок
df['ID'] = df['ID'].apply(lambda x: str(int(x)) + 'mm.txt' if not pd.isna(x) else np.nan)

# удаляем колонки которые мы не используем
columns_to_drop = ['case_qualification', 'short_version', 'Unnamed: 27']
df.drop(columns=columns_to_drop, inplace=True)

# выводим все значения из первоначального кодбука
for col in df.columns:
    print(df[col].unique())
    print(df[col].value_counts(dropna=False))

# загружаем предобработанные тексты из размеченных данных 
df2 = pd.read_csv('extracted_cases_preprocessed_marked.csv')
# объединяем с кодбуком
merged_df = pd.merge(df, df2, on='ID', how='left')

# в кодбуке есть опечатки лишние пробелы, по-разному написанные значения, приводим к общему виду
def clean_text(x):
    if isinstance(x, str):
        return x.lower().strip()
    else:
        return x

homicide_df_cleaned = merged_df.copy()

homicide_df_cleaned.iloc[:, ~homicide_df_cleaned.columns.isin(['text', 'text_prep'
                                                               )
                         ] = homicide_df_cleaned.iloc[:, ~homicide_df_cleaned.columns.isin(['text', 'text_prep'
                                                                                            ])].applymap(clean_text)

# удаляем все небинарные колонки
columns_to_drop = ['cr_settlement', 'cr_place', 'cr_place_side', 'cr_place_detailed',
                  'vi_cr_rel', 'cr_tool', 'cr_tool_how_many_attacks', 'cr_time',
                  'cr_reason', 'cr_married', 'cr_confession',
                  ]
homicide_df_cleaned.drop(columns=columns_to_drop, inplace=True)

# проверяем как называются значения колонок
for col in homicide_df_cleaned.columns:
    print(homicide_df_cleaned[col].unique())
    print(homicide_df_cleaned[col].value_counts(dropna=False))

# замена на бинарные значения, данную операцию проделываем для всех
homicide_df_cleaned['cr_sex'] = homicide_df_cleaned['cr_sex'].replace({'м': 1,
                                                                       'ж': 0})

# удаляем сырой текст
homicide_df_cleaned = homicide_df_cleaned.drop('text', axis=1)

#сохраняем в excel, далее проведена ручная разметка
homicide_df_cleaned.to_csv('marked_data.csv', index=False)

# удаляем первую строку, где указано описание колонок
marked_data = marked_data.drop(marked_data.index[0])

# необязательное действие, удаляем миссинги, появившиеся случайно
marked_data = marked_data.dropna(subset=['text_prep'])


# создаём df1 с 'ID', 'is_homicide', and 'text_prep'
df1 = marked_data[['ID', 'is_homicide', 'text_prep']]

# создаём df2 с 'ID', 'many_murderers', and 'text_prep'
df2 = marked_data[['ID', 'many_murderers', 'text_prep']]

# создаём df3 с бинарными колонками
df3 = marked_data[['ID', 'cr_sex', 'vi_sex', 'cr_other_people_around', 'cr_previous_conviction',
                   'cr_getaway', 'text_prep']]
# удаляем строки, где приговор не является убийство или несколько убийц
df3 = df3[(marked_data['is_homicide'] != 0) & (marked_data['many_murderers'] != 1)]

# создаём df4 с категориальными колонками
df4 = marked_data[['ID', 'cr_alco', 'vi_alco', 'cr_reason_clarity',
                   'cr_children', 'cr_worker', 'vi_previous_conviction', 'text_prep']]
# удаляем строки, где приговор не является убийство или несколько убийц
df4 = df4[(marked_data['is_homicide'] != 0) & (marked_data['many_murderers'] != 1)]

# в остальных случаях пропущенные значения заполняем значением "2"
df3 = df3.fillna(2)
df4 = df4.fillna(2)