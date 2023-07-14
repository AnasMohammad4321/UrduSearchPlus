import os
import json
import re

from file_operations import *
from indexing import *


def main():
    # Extract Documents.rar
    extract_rar("Documents.rar")
    
    # Read stopwords
    stopwords = read_urdu_stopwords("urdu_stopwords.txt")
    
    # Print the first 5 stopwords
    print_data(stopwords, 'stopwords')
    
    # Set the path to the documents folder
    PATH = "Documents/"
    
    # Find document IDs
    doc_ids = find_file(PATH)
    
    # Print the first 5 document IDs
    print_data(doc_ids, 'doc_ids')
    
    # Parse files and build term ID dictionary
    term_ids = {}
    term_ids = parse_files(doc_ids, PATH, term_ids, stopwords)
    
    # Print the first 10 term IDs
    print_data(term_ids, 'term_ids')
    
    # Save the term IDs and document IDs to files
    os.chdir("..")
    current_dir = os.getcwd()
    save_dict(term_ids, current_dir, "term_ids.txt", "term_id")
    save_dict(doc_ids, current_dir, "doc_ids.txt", "doc_id")
    
    # Create forward index
    forward_index = create_forward_index(doc_ids, term_ids, PATH)
    
    # Save the forward index to a file
    save_dict(forward_index, current_dir, "forward_index.txt", "forward_index")
    
    # Create inverted index
    inverted_index = create_inverted_index(forward_index)
    
    # Save the inverted index to a file
    save_dict(inverted_index, current_dir, "inverted_index.txt", "inverted_index")

if __name__ == "__main__":
    main()
