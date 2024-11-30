import sqlite3
from collections import Counter
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from itertools import islice

# Подключение к базе SQLite
def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn

# Извлечение данных из базы
def fetch_data(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM materials")  # Столбец 'content' содержит тексты
    data = cursor.fetchall()
    return [row[0] for row in data]

# Генерация n-грамм
def generate_ngrams(tokens, n=2):
    return [" ".join(pair) for pair in zip(*[islice(tokens, i, None) for i in range(n)])]

# Обработка текста с NLP (получение отдельных слов)
def process_texts_to_words(texts):
    nlp = spacy.load("ru_core_news_md")  # Загрузка модели для русского языка
    all_tokens = []
    for text in texts:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        all_tokens.extend(tokens)
    return all_tokens

# Обработка текста с NLP (получение словосочетаний)
def process_texts_to_phrases(texts):
    nlp = spacy.load("ru_core_news_md")  # Загрузка модели для русского языка
    all_bigrams = []
    for text in texts:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        bigrams = generate_ngrams(tokens, n=2)  # Генерация биграмм
        all_bigrams.extend(bigrams)
    return all_bigrams

# Анализ частоты слов или словосочетаний
def analyze_trends(items, top_n=20):
    freq = Counter(items)
    return freq.most_common(top_n)

# Визуализация облака слов или словосочетаний
def visualize_wordcloud(items, title):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(items))
    plt.figure(figsize=(10, 5))
    plt.title(title, fontsize=16)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Основной алгоритм
def main():
    db_path = "buhpulse_data.sqlite"  # Укажите путь к вашей базе
    csv_path = "22.csv"  # Укажите путь к вашему CSV
    use_csv = True  # Выбор источника данных

    if use_csv:
        print("Чтение данных из CSV...")
        try:
            data = pd.read_csv(csv_path)
            texts = data["text"].dropna().tolist()
        except Exception as e:
            print(f"Ошибка чтения CSV: {e}")
            return
    else:
        print("Подключение к базе данных...")
        conn = connect_to_db(db_path)
        try:
            texts = fetch_data(conn)
        finally:
            conn.close()

    if not texts:
        print("Нет данных для обработки.")
        return

    print("Обработка текста для облака слов...")
    words = process_texts_to_words(texts)

    print("Обработка текста для облака словосочетаний...")
    phrases = process_texts_to_phrases(texts)

    print("Анализ трендов для слов...")
    word_trends = analyze_trends(words)
    print("Топ слов:", word_trends)

    print("Анализ трендов для словосочетаний...")
    phrase_trends = analyze_trends(phrases)
    print("Топ словосочетаний:", phrase_trends)

    print("Визуализация облака слов...")
    visualize_wordcloud(words, title="Облако слов")

    print("Визуализация облака словосочетаний...")
    visualize_wordcloud(phrases, title="Облако словосочетаний")

if __name__ == "__main__":
    main()
