from flask import Flask, render_template, request, jsonify
import json
import math
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load term_id.json and inverted_index.json outside of the functions
with open('term_id.json', 'r', encoding='utf-8') as term_id_file:
    term_id_data = json.load(term_id_file)

with open('inverted_index.json', 'r', encoding='utf-8') as inverted_index_file:
    inverted_index_data = json.load(inverted_index_file)

def preprocess_query(query):
    tokens = word_tokenize(query)
    with open('urdu_stopwords.txt', 'r', encoding='utf-8') as stopwords_file:
        stop_words = stopwords_file.read().splitlines()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def calculate_tfidf(query, inverted_index, total_documents):
    tfidf_scores = {}
    for term in query:
        if term in inverted_index:
            df = len(inverted_index[term])
            idf = math.log10(total_documents / df) if df > 0 else 0
            tfidf_scores[term] = idf
    return tfidf_scores

def retrieve_documents(tfidf_scores, inverted_index):
    relevant_documents = set()
    for term, idf in tfidf_scores.items():
        if term in inverted_index:
            relevant_documents.update(inverted_index[term])
    return relevant_documents

# Assign the loaded data to global variables
term_id = term_id_data
inverted_index = inverted_index_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        query_text = request.form.get('query')
        preprocessed_query = preprocess_query(query_text)
        term_id_inverted = {v: k for k, v in term_id.items()}
        query_term_ids = [term_id_inverted.get(term) for term in preprocessed_query if term in term_id_inverted]
        total_documents = 5771

        tfidf_scores = {}
        for term_id_val in query_term_ids:
            if term_id_val in inverted_index:
                for doc_id, positions in inverted_index[term_id_val].items():
                    tf = len(positions)
                    df = len(inverted_index[term_id_val])
                    idf = math.log10(total_documents / df) if df > 0 else 0
                    total_terms_in_doc = len(inverted_index[term_id_val][doc_id])
                    normalized_tf = tf / total_terms_in_doc if total_terms_in_doc > 0 else 0
                    tfidf_score = normalized_tf * idf
                    tfidf_scores.setdefault(doc_id, 0)
                    tfidf_scores[doc_id] += tfidf_score

        sorted_documents = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        # Returning JSON response for simplicity, you might want to render a template or HTML page
        return jsonify(sorted_documents)

if __name__ == "__main__":
    app.run(debug=True)
