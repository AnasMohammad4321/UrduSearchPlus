# UrduSearchPlus

Discover the Beauty of Urdu with Enhanced Search Capabilities

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/AnasMohammad4321/UrduSearchPlus/blob/main/LICENSE)

## Description
UrduSearchPlus is a powerful search engine designed to facilitate searching and indexing of Urdu text. It offers advanced features such as natural language processing, stopword removal, and indexing algorithms for efficient searching. With UrduSearchPlus, you can explore and analyze Urdu documents with ease.

## Features

### Search Engine:
- Users can perform searches in Urdu.
- Search is powered by TF-IDF scoring and an inverted index.

### User Interface:
- Simple and responsive web interface for search.
- Stylish design with Font Awesome icons.

### Document Processing:
- Extraction of documents from a compressed file on startup.
- Handling of Urdu stopwords for improved search accuracy.

### Backend Processing:
- Flask-based backend for handling search requests.
- Generation of term IDs and document IDs.

### Result Display:
- Dynamic display of search results without page reload.
- Results include relevant information like document ID, score, and document name.

### File Operations:
- Reading stopwords and finding document IDs.
- Saving term IDs and document IDs to text files.

### Indexing:
- Construction of a term ID dictionary, forward index, and inverted index.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnasMohammad4321/UrduSearchPlus.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the Flask application:
   ```bash
   python app.py
2. Open your web browser and go to http://localhost:5000 to access UrduSearchPlus.

## License
This project is licensed under the MIT License.
