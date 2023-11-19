from flask import Flask, render_template, request, jsonify
import json
import math
from nltk.tokenize import word_tokenize
import os

app = Flask(__name__)

with open('doc_id.json', 'r', encoding='utf-8') as doc_id_file:
    doc_id_data = json.load(doc_id_file)

with open('term_id.json', 'r', encoding='utf-8') as term_id_file:
    term_id_data = json.load(term_id_file)

with open('inverted_index.json', 'r', encoding='utf-8') as inverted_index_file:
    inverted_index_data = json.load(inverted_index_file)

doc_id = doc_id_data
term_id = term_id_data
inverted_index = inverted_index_data

def preprocess_query(query):
    """
    Preprocesses the query by tokenizing and removing stopwords.

    Args:
        query (str): The input query.

    Returns:
        list: List of filtered tokens.
    """
    tokens = word_tokenize(query)
    with open('urdu_stopwords.txt', 'r', encoding='utf-8') as stopwords_file:
        stop_words = stopwords_file.read().splitlines()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def calculate_tfidf_score(tf, df, total_documents, total_terms_in_doc):
    """
    Calculates the TF-IDF score for a term.

    Args:
        tf (int): Term frequency.
        df (int): Document frequency.
        total_documents (int): Total number of documents.
        total_terms_in_doc (int): Total terms in the document.

    Returns:
        float: TF-IDF score.
    """
    idf = math.log10(total_documents / df) if df > 0 else 0
    normalized_tf = tf / total_terms_in_doc if total_terms_in_doc > 0 else 0
    tfidf_score = normalized_tf * idf
    return tfidf_score

def calculate_tfidf_scores(query_term_ids, inverted_index, total_documents):
    """
    Calculates TF-IDF scores for the given query term IDs.

    Args:
        query_term_ids (list): List of query term IDs.
        inverted_index (dict): Inverted index data structure.
        total_documents (int): Total number of documents.

    Returns:
        dict: Dictionary of document IDs and corresponding TF-IDF scores.
    """
    tfidf_scores = {}

    for term_id in query_term_ids:
        if term_id in inverted_index:
            for doc_id, positions in inverted_index[term_id].items():
                tf = len(positions)
                df = len(inverted_index[term_id])
                total_terms_in_doc = len(inverted_index[term_id][doc_id])

                tfidf_score = calculate_tfidf_score(tf, df, total_documents, total_terms_in_doc)

                tfidf_scores.setdefault(doc_id, 0)
                tfidf_scores[doc_id] += tfidf_score

    return tfidf_scores

def open_and_print_top_documents(sorted_documents, doc_id_mapping):
    """
    Retrieves and formats the top documents based on TF-IDF scores.

    Args:
        sorted_documents (list): List of tuples containing document IDs and TF-IDF scores.
        doc_id_mapping (dict): Mapping of document IDs to document names.

    Returns:
        list: List of dictionaries representing top search results.
    """
    results = []

    documents_folder = "Documents"  # Replace with the actual path to your Documents folder

    for rank, (doc_id, score) in enumerate(sorted_documents[:10], start=1):
        doc_name = doc_id_mapping.get(doc_id, f"Unknown Document {doc_id}")
        doc_path = os.path.join(documents_folder, doc_name)

        try:
            with open(doc_path, 'r', encoding='utf-8') as doc_file:
                first_line = doc_file.readline().strip()
                result = {
                    'Rank': rank,
                    'Document Name': doc_name,
                    'TF-IDF Score': score,
                    'First Line': first_line
                }
                results.append(result)
        except FileNotFoundError:
            print(f"Document not found: {doc_name}")

    return results

@app.route('/')
def index():
    """
    Renders the index.html template.

    Returns:
        str: Rendered HTML page.
    """
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """
    Handles the search functionality and returns JSON response.

    Returns:
        str: JSON response containing search results.
    """
    if request.method == 'POST':
        query_text = request.form.get('query')
        preprocessed_query = preprocess_query(query_text)
        term_id_inverted = {v: k for k, v in term_id.items()}
        query_term_ids = [term_id_inverted.get(term) for term in preprocessed_query if term in term_id_inverted]
        total_documents = 5771

        tfidf_scores = calculate_tfidf_scores(query_term_ids, inverted_index, total_documents)

        sorted_documents = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

        # Get the output to be displayed in index.html
        result_output = open_and_print_top_documents(sorted_documents, doc_id)

        return jsonify(result_output)

if __name__ == "__main__":
    app.run(debug=True)
