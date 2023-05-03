# Python Search Engine

## Purpose
Collects documents from a curated list of URLs to index and search querys on. Each URL pertains to a specific topic and by training a machine learning algorithm on the collected documents the program can predict the topic of a given URL. 

## Starting

Before starting, you need to have [Git](https://git-scm.com) and [Python](https://www.python.org/downloads/) installed.

```bash
# Clone the Repository

$ git clone https://github.com/mr-rjh3/python-search-engine

# Access the project directory

$ cd python-search-engine

# Install the requirements

$ pip install -r requirements.txt

# Run searchengine.py

$ python ./searchengine.py
```

## Usage
Once started the program will offer 6 options:

```
Select an option: 

        1 - Collect new documents.
        2 - Index documents.      
        3 - Search for a query.   
        4 - Train ML classifier.  
        5 - Predict a link.       
        6 - Quit.

Enter your choice: 
```


### Collect new documents

- Collect new documents will scrape the URLs listed in 'sources.txt' and store the text content of that page with stopwords removed in it's given topic directory. It will continue this for all outgoing links on the page that stay in it's root domain to a maximum depth of 2.

### Index documents

- Index documents will create an inverted index of the collected documents from the previous option.

### Search for a query

- Search for a query will take in a query as input and search for the documents with the greatest cosine similarity using the inverted index.

### Train ML classifier

- Train ML classifier will train an SVM machine learning classifier on the gathered documents using their topics as the label. The scores and confusion matrix will be shown to the user once training is complete.

### Predict a link

- Predict a link will take in a URL as user input and using the classifier trained in the previous option, predict the topic of the given link along with the confidence of the prediction.

### Quit

- Quit will simply exit the program.













