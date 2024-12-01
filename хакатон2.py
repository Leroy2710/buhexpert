import sqlite3
import re
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
# Функция для подключения к базе данных
def connect_to_db(db_name):
    conn = sqlite3.connect(db_name)
    return conn

# Функция для извлечения данных из базы данных
def fetch_data(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, date, user_id, text FROM {table_name}")
    rows = cursor.fetchall()
    return rows

# Функция для объединения комментариев, занимающих несколько строк
def merge_multiline_comments(comments):
    merged_comments = []
    current_comment = None
    for comment in comments:
        if comment[0] is not None:  # Если у комментария есть id
            if current_comment:
                merged_comments.append(current_comment)
            current_comment = comment
        else:
            current_comment = (current_comment[0], current_comment[1], current_comment[2], current_comment[3] + ' ' + comment[3])
    if current_comment:
        merged_comments.append(current_comment)
    return merged_comments

# Функция для подготовки данных для обучения модели
def prepare_data_for_spam_detection(comments, spam_keywords):
    data = []
    labels = []
    for comment in comments:
        text = comment[3].lower()
        label = 1 if any(keyword in text for keyword in spam_keywords) else 0
        data.append(text)
        labels.append(label)
    return data, labels

# Функция для обучения модели
def train_spam_model(data, labels, model_path="spam_model.pkl", vectorizer_path="vectorizer.pkl"):
    # Создание векторизатора
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    y = labels

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Предсказание и вывод метрик
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Сохранение модели и векторизатора
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_path, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

    return model, vectorizer

# Функция для фильтрации спама с использованием модели
def filter_spam_with_model(model, vectorizer, comments):
    data = [comment[3].lower() for comment in comments]
    X = vectorizer.transform(data)
    predictions = model.predict(X)
    filtered_comments = [comment for comment, prediction in zip(comments, predictions) if prediction == 0]
    return filtered_comments

# Функция для фильтрации комментариев по ключевым словам
def filter_comments_by_keywords(comments, keywords):
    filtered_comments = [comment for comment in comments if any(re.search(r'\b' + keyword, comment[3].lower()) for keyword in keywords)]
    return filtered_comments

# Функция для записи комментариев в файл
def write_comments_to_file(comments, filename):
    # Убираем запятые из текста
    comments = [(id, date, user_id, text.replace(',', ''),) for id, date, user_id, text in comments]
    # Заключаем все данные в кавычки
    with open(filename, mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['id', 'date', 'user_id', 'text'])
        for comment in comments:
            writer.writerow(comment)

# Основная функция для выполнения всех задач
def main():
    db_name = 'buhpulse_data.sqlite'  # Имя вашей базы данных
    conn = connect_to_db(db_name)

    table_names = [
        'messages_10', 'messages_11', 'messages_12', 'messages_13',
        'messages_2', 'messages_3', 'messages_4', 'messages_5',
        'messages_6', 'messages_7', 'messages_8', 'messages_9'
    ]

    all_comments = []
    for table_name in table_names:
        comments = fetch_data(conn, table_name)
        all_comments.extend(comments)

    spam_keywords = ['купить', 'продать', 'бесплатно']
    data, labels = prepare_data_for_spam_detection(all_comments, spam_keywords)
    model, vectorizer = train_spam_model(data, labels)

    filtered_comments = filter_spam_with_model(model, vectorizer, all_comments)

    accounting_keywords = [
        r'финанс', r'бухгалтер', r'аудит', r'проверк', r'ревизи', r'контроль', r'инспекци',
        r'отчетност', r'налог', r'налогообложени', r'услуг'
    ]
    comments = filter_comments_by_keywords(filtered_comments, accounting_keywords)

    merged_comments = merge_multiline_comments(comments)

    # Запись комментариев в файл
    output_filename = 'merged_comments.csv'
    write_comments_to_file(merged_comments, output_filename)
    print(f"Комментарии записаны в файл {output_filename}")

    conn.close()

if __name__ == "__main__":
    main()
