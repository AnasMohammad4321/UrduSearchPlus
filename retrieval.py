import nltk
from nltk.tokenize import word_tokenize
import math
import json
import os


def preprocess_query(query):
    """Preprocesses an Urdu query.

    Args:
        query (str): The input query in Urdu.

    Returns:
        list: A list of preprocessed tokens after tokenization and stop-word removal.
    """
    tokens = word_tokenize(query)
    with open('urdu_stopwords.txt', 'r', encoding='utf-8') as stopwords_file:
        stop_words = stopwords_file.read().splitlines()
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens


def calculate_tfidf_score(tf, df, total_documents, total_terms_in_doc):
    """Calculates TF-IDF score for a term in a document.

    Args:
        tf (int): Term frequency.
        df (int): Document frequency.
        total_documents (int): Total number of documents in the collection.
        total_terms_in_doc (int): Total number of terms in the document.

    Returns:
        float: TF-IDF score.
    """
    idf = math.log10(total_documents / df) if df > 0 else 0
    normalized_tf = tf / total_terms_in_doc if total_terms_in_doc > 0 else 0
    tfidf_score = normalized_tf * idf

    return tfidf_score


def calculate_tfidf_scores(query_term_ids, inverted_index, total_documents):
    """Calculates TF-IDF scores for terms in an Urdu query.

    Args:
        query_term_ids (list): Term IDs present in the preprocessed query.
        inverted_index (dict): The inverted index containing document frequencies for terms.
        total_documents (int): The total number of documents in the collection.

    Returns:
        dict: A dictionary containing TF-IDF scores for terms present in the query.
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
    """Prints the first line of the top 10 ranked documents.

    Args:
        sorted_documents (list): List of tuples containing (doc_id, TF-IDF score).
        doc_id_mapping (dict): Mapping of doc_id to document names.
    """
    documents_folder = "Documents"  # Replace with the actual path to your Documents folder

    for rank, (doc_id, score) in enumerate(sorted_documents[:10], start=1):
        doc_name = doc_id_mapping.get(doc_id, f"Unknown Document {doc_id}")
        doc_path = os.path.join(documents_folder, doc_name)

        try:
            with open(doc_path, 'r', encoding='utf-8') as doc_file:
                first_line = doc_file.readline().strip()
                print(f"Rank: {rank}, Document Name: {doc_name}, TF-IDF Score: {score}, First Line: {first_line}")
        except FileNotFoundError:
            print(f"Document not found: {doc_name}")


def main():
    # Read the query from query.txt
    with open('query.txt', 'r', encoding='utf-8') as query_file:
        query_text = query_file.read()

    # Read doc_id.json
    with open('doc_id.json', 'r', encoding='utf-8') as doc_id_file:
        doc_id = json.load(doc_id_file)

    # Read term_id.json
    with open('term_id.json', 'r', encoding='utf-8') as term_id_file:
        term_id = json.load(term_id_file)

    # Read inverted_index.json
    with open('inverted_index.json', 'r', encoding='utf-8') as inverted_index_file:
        inverted_index = json.load(inverted_index_file)

    # Preprocess the query
    preprocessed_query = preprocess_query(query_text)

    # Map term IDs from terms in the preprocessed query
    term_id_inverted = {v: k for k, v in term_id.items()}
    query_term_ids = [term_id_inverted.get(term) for term in preprocessed_query if term in term_id_inverted]
    total_documents = 5771

    # Calculate TF-IDF scores
    tfidf_scores = calculate_tfidf_scores(query_term_ids, inverted_index, total_documents)

    # Sort the documents based on their TF-IDF scores in descending order
    sorted_documents = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    # Print the first line of the top 10 ranked documents
    open_and_print_top_documents(sorted_documents, doc_id)


if __name__ == "__main__":
    main()
