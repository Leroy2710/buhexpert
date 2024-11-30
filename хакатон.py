import sqlite3
import re
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from collections import Counter

# Функция для подключения к базе данных
def connect_to_db(db_name):
    conn = sqlite3.connect(db_name)
    return conn

# Функция для извлечения данных из базы данных
def fetch_data(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, date, user_id, text FROM messages_10")
    rows = cursor.fetchall()
    return rows

# Функция для фильтрации спама
def filter_spam(comments, spam_keywords):
    filtered_comments = [comment for comment in comments if not any(keyword in comment[3] for keyword in spam_keywords)]
    return filtered_comments

# Функция для подготовки данных
def prepare_data(comments, accounting_keywords):
    data = []
    labels = []
    for comment in comments:
        text = comment[3].lower()
        label = 1 if any(re.search(r'\b' + keyword, text) for keyword in accounting_keywords) else 0
        data.append(text)
        labels.append(label)
    return data, labels

# Функция для обучения модели
def train_model(data, labels):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model, vectorizer

# Функция для классификации новых комментариев
def classify_comments(model, vectorizer, comments):
    data = [comment[3].lower() for comment in comments]
    X = vectorizer.transform(data)
    predictions = model.predict(X)
    relevant_comments = [comment for comment, prediction in zip(comments, predictions) if prediction == 1]
    return relevant_comments

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

# Функция для кластеризации комментариев
def cluster_comments(comments, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([comment[3].lower() for comment in comments])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    clustered_comments = [(comment[0], comment[1], comment[2], comment[3], label) for comment, label in zip(comments, labels)]
    return clustered_comments, vectorizer, kmeans

# Функция для назначения имен кластерам
def name_clusters(clustered_comments, vectorizer, kmeans, top_n_words=5):
    cluster_names = {}
    for cluster_id in range(kmeans.n_clusters):
        cluster_comments = [comment[3] for comment in clustered_comments if comment[4] == cluster_id]
        cluster_text = ' '.join(cluster_comments)
        word_freq = Counter(cluster_text.split())
        top_words = [word for word, _ in word_freq.most_common(top_n_words)]
        cluster_names[cluster_id] = ' '.join(top_words)
    return cluster_names

# Функция для записи комментариев в файл
def write_comments_to_file(comments, cluster_names, filename):
    # Убираем запятые из текста
    comments = [(id, date, user_id, text.replace(',', ''), label) for id, date, user_id, text, label in comments]
    # Заключаем все данные в кавычки
    with open(filename, mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['id', 'date', 'user_id', 'text', 'cluster', 'cluster_name'])
        for comment in comments:
            cluster_name = cluster_names.get(comment[4], 'Unknown')
            writer.writerow(comment + (cluster_name,))

# Основная функция для выполнения всех задач
def main():
    db_name = 'buhpulse_data.sqlite'  # Имя вашей базы данных
    conn = connect_to_db(db_name)
    comments = fetch_data(conn)

    spam_keywords = ['купить', 'продать', 'бесплатно']
    comments = filter_spam(comments, spam_keywords)

    accounting_keywords = [
        r'финанс', r'бухгалтер', r'аудит', r'проверк', r'ревизи', r'контроль', r'инспекци',
        r'отчетност', r'налог', r'налогообложени', r'услуг'
    ]
    data, labels = prepare_data(comments, accounting_keywords)
    model, vectorizer = train_model(data, labels)

    relevant_comments = classify_comments(model, vectorizer, comments)
    print(f"Найдено {len(relevant_comments)} комментариев, связанных с бухгалтерскими услугами.")

    merged_comments = merge_multiline_comments(relevant_comments)

    clustered_comments, vectorizer, kmeans = cluster_comments(merged_comments, num_clusters=5)
    cluster_names = name_clusters(clustered_comments, vectorizer, kmeans)

    # Запись комментариев в файл
    output_filename = 'clustered_comments.csv'
    write_comments_to_file(clustered_comments, cluster_names, output_filename)
    print(f"Комментарии записаны в файл {output_filename}")

    conn.close()

if __name__ == "__main__":
    main()
