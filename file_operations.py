import os
import re
import json
import nltk
from tqdm import tqdm

def extract_rar(file_path):
    """
    Extracts a RAR file to the given destination path.

    Args:
        file_path (str): The path to the RAR file.
        destination_path (str): The path to the destination folder.
    """
    os.system(f"unrar x {file_path}")


def natural_sort_key(s):
    """
    Key function for natural sorting.

    Args:
        s (str): The string to generate the natural sort key for.

    Returns:
        list: The list of values that can be used for natural sorting.

    Example:
        The natural sort key for ['file1.txt', 'file10.txt', 'file2.txt'] would be [file, 1, .txt], [file, 10, .txt], [file, 2, .txt].
    """
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', s)]

def find_file(folder_path):
    """
    Finds all files in a given folder path, then adds them to a dictionary
    of the form {file_number: file_name}.

    Args:
        folder_path (str): The path to the folder that contains the files.

    Returns:
        dict: A dictionary of the form {file_number: file_name}.
    """
    file_dict = {}

    file_list = sorted(os.listdir(folder_path), key=natural_sort_key)

    for i, file_name in enumerate(file_list, start=1):
        file_dict[i] = file_name

    return file_dict

from tqdm import tqdm

def parse_files(doc_ids, path, term_id, stopwords):
    """
    Parse files from the given document IDs and path, and build a dictionary of terms and term IDs.

    Args:
        doc_ids (dict): A dictionary containing the document IDs.
        path (str): The path to the folder containing the files.
        term_id (dict): The existing dictionary of terms and term IDs to be updated.
        stopwords (list): A list of stopwords.

    Returns:
        dict: The updated dictionary of terms and term IDs.
    """
    os.chdir(path)

    existing_terms = set(term_id.values())
    html_tag_regex = re.compile(r'<.*?>')
    punctuation_regex = re.compile(r'[^\w\s]')

    for file_name in tqdm(doc_ids.values(), desc="Parsing files", unit="file"):
        with open(file_name, "r", encoding="utf-8") as f:
            # Read the file
            text = f.read()

            # Remove HTML tags
            text = html_tag_regex.sub('', text)

            # Remove punctuation
            text = punctuation_regex.sub('', text)

            # Remove non-letters
            text = re.sub("[^آ-ی]", " ", text)

            # Remove extra spaces
            text = re.sub(" +", " ", text)

            # Tokenize
            tokens = nltk.word_tokenize(text)

            # Remove stopwords
            tokens = [token for token in tokens if token not in stopwords]

            # Update the dictionary of terms and term IDs
            for token in tokens:
                if token not in existing_terms:
                    term_id[len(term_id) + 1] = token

                    # Add the newly encountered term to the existing terms set
                    existing_terms.add(token)

    return term_id

def print_data(data, data_type, items=5):
    """
    Print a specified number of key-value pairs from the given data based on the data type.

    Args:
        data (dict or list): The dictionary or list containing the data.
        data_type (str): The type of data ('term_ids', 'stopwords', 'doc_ids') to print.
        items (int): The number of key-value pairs to print. Defaults to 10.
    """
    count = 0
    if data_type == 'term_ids':
        for key, value in data.items():
            print(f"Term ID: {key}  ===>  Term: {value}")
            count += 1
            if count == items:
                break
    elif data_type == 'stopwords':
        for word in data:
            print(word)
            count += 1
            if count == items:
                break
    elif data_type == 'doc_ids':
        for key, value in data.items():
            print(f"Document ID: {key}  ===>  File Name: {value}")
            count += 1
            if count == items:
                break
    elif data_type == 'forward_index':
        for key, value in data.items():
            print(f"Document ID: {key}")
            for term, positions in value.items():
                print(f"    Term: {term}")
                print(f"    Positions: {positions}")
            count += 1
            if count == items:
                break
    elif data_type == 'inverted_index':
        for term, docs in data.items():
            print(f"Term: {term}")
            for docid, positions in docs.items():
                print(f"    Document ID: {docid}")
                print(f"    Positions: {positions}")
            count += 1
            if count == items:
                break
    else:
        print("Invalid data type. Please provide valid data type: 'term_ids', 'stopwords', 'doc_ids'")

def read_urdu_stopwords(path):
    """Reads the stopwords from the given file path.
    
    Args:
        path (str): The path to the file containing the stopwords.
    
    Returns:
        list: A list of stopwords.
    
    """
    
    with open(path, "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()
        
    return list(stopwords)

def save_dict(dict, path, file_name, type):
    """
    Save a dictionary to a file in JSON format.

    Args:
        dict (dict): The dictionary to be saved.
        path (str): The path to the folder where the file will be saved.
        file_name (str): The name of the file to be saved.
        type (str): The type of dictionary being saved. Options: 'doc_id', 'term_id', 'forward_index', 'inverted_index'.
    """
    os.chdir(path)  # Set the current working directory to the given path

    if type == "doc_id":
        # Save the dictionary as doc_id.json
        with open("doc_id.json", "w", encoding="utf-8") as f:
            json.dump(dict, f, ensure_ascii=False)
            f.close()
            
    elif type == "term_id":
        # Save the dictionary as term_id.json
        with open("term_id.json", "w", encoding="utf-8") as f:
            json.dump(dict, f, ensure_ascii=False)
            f.close()
            
    elif type == "forward_index":
        # Save the dictionary as forward_index.json
        with open("forward_index.json", "w", encoding="utf-8") as f:
            json.dump(dict, f, ensure_ascii=False)
            f.close()
            
    elif type == "inverted_index":
        # Save the dictionary as inverted_index.json
        with open("inverted_index.json", "w", encoding="utf-8") as f:
            json.dump(dict, f, ensure_ascii=False)
            f.close()
            
    else:
        print("Invalid type. Please provide valid type: 'doc_id', 'term_id', 'forward_index', 'inverted_index'")
        
