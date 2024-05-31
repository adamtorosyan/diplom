library(data.table)  # Импорт библиотеки data.table для работы с таблицами данных
library(R.utils)  # Импорт библиотеки R.utils для работы с утилитами, такими как gunzip

set.seed(1)  # Установка случайного начального значения для воспроизводимости результатов

# Чтение данных из CSV файла в таблицу данных
hom <- fread("intermediate_data/rospravosudie_homicides_2011.csv")
hom[, id := 1:nrow(hom)]  # Добавление столбца с идентификаторами строк

n_of_verdicts <- nrow(hom)  # Количество приговоров в таблице

hom_random_full <- sample(n_of_verdicts)  # Перемешивание индексов строк

hom_sample_10pct <- hom[hom_random_full[1:(n_of_verdicts / 10)], ]  # Выборка 10% строк из таблицы

# Цикл обработки каждой строки в выборке
for (i in 1:nrow(hom_sample_10pct)) {
  # Копирование файла в директорию выборки
  invisible(file.copy(
    from = hom_sample_10pct$path[i],  # Исходный путь файла
    to = "intermediate_data/verdicts_sample_10pct/",  # Директория назначения
    overwrite = TRUE  # Перезапись файла, если он уже существует
  ))
  
  file_name <- basename(hom_sample_10pct$path[i])  # Извлечение имени файла из полного пути
  
  new_file_path <- file.path("intermediate_data/verdicts_sample_10pct/", file_name)  # Формирование нового пути к файлу
  
  # Распаковка файла
  gunzip(new_file_path, remove = TRUE, overwrite = TRUE)  # Распаковка файла и удаление исходного архива
  
  new_file_path_xml <- substr(new_file_path, 0, nchar(new_file_path) - 3)  # Формирование пути к распакованному XML файлу
  
  new_file_path_html <- file.path("intermediate_data/verdicts_sample_10pct/", paste0(i, ".html"))  # Формирование пути к HTML файлу с индексом
  
  # Чтение содержимого распакованного XML файла
  verdict_raw_text <- paste0(readLines(new_file_path_xml, encoding = "utf-8"), collapse = "")[[1]]  # Чтение всех строк файла и объединение их в одну строку
  
  library(stringr)  # Импорт библиотеки stringr для работы с текстовыми строками
  verdict_body <- stringr::str_match(verdict_raw_text, "CDATA.+") %>% 
    substr(., 7, nchar(.) - 10)  # Извлечение текста приговора из CDATA секции
  
  # Создание HTML содержимого
  verdict_html <- paste0(
    '<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>№', i, ' ID ', hom_sample_10pct$id[i], '</title>
  </head>
  <body>',
    verdict_body,
    '
  </body>
</html>'
  )
  
  # Запись HTML содержимого в файл
  writeLines(verdict_html, new_file_path_html)  # Запись HTML содержимого в файл
  invisible(file.remove(new_file_path_xml))  # Удаление распакованного XML файла
}