import os
import json
import re
import nltk
nltk.download('punkt')

from file_operations import *
from indexing import *

def main():
    extract_rar("Documents.rar")
    
    stopwords = read_urdu_stopwords("urdu_stopwords.txt")
    print_data(stopwords, 'stopwords')
    
    PATH = "Documents/"
    doc_ids = find_file(PATH)
    print_data(doc_ids, 'doc_ids')
    
    term_ids = {}
    term_ids = parse_files(doc_ids, PATH, term_ids, stopwords)
    print_data(term_ids, 'term_ids')
    
    os.chdir("..")
    current_dir = os.getcwd()
    save_dict(term_ids, current_dir, "term_ids.txt", "term_id")
    save_dict(doc_ids, current_dir, "doc_ids.txt", "doc_id")
    
    forward_index = create_forward_index(doc_ids, term_ids, PATH)
    save_dict(forward_index, current_dir, "forward_index.txt", "forward_index")
    
    inverted_index = create_inverted_index(forward_index)
    save_dict(inverted_index, current_dir, "inverted_index.txt", "inverted_index")

if __name__ == "__main__":
    main()
