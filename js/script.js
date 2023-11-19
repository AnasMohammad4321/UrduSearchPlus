document.getElementById('search-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);
  
    fetch('/search', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      const resultsContainer = document.getElementById('search-results');
      resultsContainer.innerHTML = '<h2>Search Results:</h2>';
  
      data.forEach(result => {
        const { 'Document ID': doc_id, 'TF-IDF Score': score, 'Rank': rank, 'Document Name': doc_name, 'First Line': first_line } = result;
        const resultItem = `<p>Rank: ${rank}, Document ID: ${doc_id}, Document Name: ${doc_name}, TF-IDF Score: ${score}, First Line: ${first_line}</p>`;
        resultsContainer.innerHTML += resultItem;
      });
    })
    .catch(error => console.error('Error:', error));
  });
  