import  collections
import nltk
from tqdm import tqdm
import os

def create_forward_index(doc_id, term_id, path):
    """
    Create a forward index from the given document IDs, term IDs, and folder path.

    Args:
        doc_id (dict): A dictionary containing the document IDs.
        term_id (dict): A dictionary containing the term IDs.
        path (str): The path to the folder containing the files.

    Returns:
        dict: The forward index containing the document-term positions.
    """
    forward_index = collections.defaultdict(lambda: collections.defaultdict(list))
    term_to_termid = {v: k for k, v in term_id.items()}
    os.chdir(path)

    pbar = tqdm(doc_id.items(), desc="Processing documents", unit="document")
    for docid, filename in pbar:
        with open(filename, "r", encoding='utf-8') as file:
            text = file.read()
            tokens = nltk.word_tokenize(text)
            position = 1

            for term in tokens:
                if term in term_to_termid:
                    forward_index[docid][term_to_termid[term]].append(position)
                    position += 1

        pbar.set_postfix({"Processed": docid})
    
    pbar.close()

    return forward_index


def create_inverted_index(forward_index):
    """
    Create an inverted index from the given forward index.

    Args:
        forward_index (dict): The forward index containing the document-term positions.

    Returns:
        dict: The inverted index containing the term-document positions.
    """
    inverted_index = collections.defaultdict(lambda: collections.defaultdict(list))

    for docid, terms in tqdm(forward_index.items(), desc="Processing documents", unit="document"):
        for termid, positions in terms.items():
            inverted_index[termid][docid].extend(positions)

    return inverted_index
