# Import libraries
import json
import os.path
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from hashlib import blake2b
from nltk.corpus import stopwords
import fuzzy

# nltk libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sklearn training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Sklearn Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Sklearn metrics
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

# pickle for saving the model
import pickle

# visualization
import matplotlib.pyplot as plt

MAXDEPTH = 2

# LABELS
ENTERTAINMENT = 0
LIFESTYLE = 1
TECHNOLOGY = 2

class printColors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def removeStopwords(text):
    # remove stopwords from text
    stop_words = set(stopwords.words('english'))
    filtered_text = ' '.join([w for w in text if not w in stop_words])
    return filtered_text

def getContentFromSoup(soup):
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    try:
        text = soup.body.get_text(separator=' ')
    except:
        text = soup.get_text(separator=' ')
    content = removeStopwords(text.split())
    return content

def scrapeURL(URL, maxdepth, topic, depth=0):
    if(depth >= maxdepth):
        # We have reached the maximum depth so we will return
        return
    # Find the hash of the URL
    H = blake2b(bytes(URL, encoding='utf-8')).hexdigest()

    # check if H.txt exists (H is the hash of the URL)
    if(os.path.isfile(f"data/{topic}/{H}.txt")):
       return

    # the file does not exist so we will continue scraping the URL
    
    # try to get the page / URL to scrape
    try:
        page = requests.get(URL)
    except:
        # print("Error: Could not scrape URL: ", URL)
        return
    
    # Write to the log file
    with open("crawler.log", "a") as f:
        # <topic, link’s URL, Hash value of URL, date>
        f.write(f"<{topic}, {URL}, {H}, {datetime.now()}, {page.status_code}>\n")
    
    if(page.status_code != 200):
        # print("Error: Could not scrape URL: ", URL)
        return

    # Initialize the BeautifulSoup object
    try:
        soup = BeautifulSoup(bytes(page.text, encoding="utf-8"), "html.parser")
    except Exception as e: # catch all exceptions
        print(e)
        print("Could not parse page. skipping...")
        return

    # Extract all links from the page that share the same root domains
    links = soup.find_all("a", href=True)

    # kill all script and style elements
    content = getContentFromSoup(soup)

    with open(f"data/{topic}/" + H + ".txt", "w", encoding="utf-8") as f:
        f.write(content)

    for link in links:
        # Recursively call the function to scrape the links
        # print(link)
        if("http" in link['href'] and URL.split("/")[2] in link['href']): # check if the link is from the same root domain
            scrapeURL(link['href'], maxdepth, topic, depth+1)
        elif("http" not in link['href'] and len(link['href']) > 1 and link['href'][0] == '/'): # check if the link is a relative link
            scrapeURL('/'.join(URL.split("/")[:3]) + link['href'], maxdepth, topic, depth+1) # add the relative link to the root domain
        else: # the link is not from the same root domain
            # print("Skipping URL: ", link['href'])
            continue
            
def collectDocuments():
    # crawl each link in sources.txt
    if(not os.path.isfile("sources.txt")):
        raise("sources.txt not found.")
    
    # read the sources.txt file and return the list of links
    with open("sources.txt", "r") as f:
        links = f.readlines()

    for link in links:
        if(link[0] == "#"):
            # This is a comment
            continue
        topic, URL = link.split(",")
        URL = URL.strip()
        topic = topic.strip()
        print("Scraping URL: ", URL)
        scrapeURL(URL, MAXDEPTH, topic)
    return

def indexDocuments():

    index = {}
    soundex = fuzzy.Soundex(4)

    # For each topic we chose,,
    for category in os.listdir("data"):

        # Go into each file in one topic at a time
        if(os.listdir(f"data/{category}") == []):
            continue
        for filename in os.listdir(f"data/{category}"):

            # If the file is the .gitIgnore, continue
            if filename[0] == ".":
                continue

            # Open file and read the article into a string
            with open(f"data/{category}/{filename}", "r", encoding="utf-8") as f:
                article = f.read()

            # Create hash of the article
            h = filename.split('.')[0]

            # Tokenize the article
            tokenized = word_tokenize(article)
            
            # For each token in the article..
            for token in tokenized:
                if ("|" in token):
                    continue 
                # lowercase the token
                token = token.lower()
                    # See if the token exists in the index 
                if(token in index.keys()):
                    # If the token does exist, increment the frequency
                    # of the token in this specific article 
                    if(h in index[token][1].keys()):
                        index[token][1][h][0] += 1
                    # If the hash does not exist, add it to the dictionary
                    else:
                        index[token][1][h] = [1, category]
                # If the token does not exist, add it to the dictionary
                else:
                    index[token] = [soundex(token.encode('utf-8')), {h: [1, category]}]
        
    if(index == {}):
        print("Error: No documents to index. Please run Collect new documnets first.")
        return


    # Open text file to write inverted index
    with open("invertedindex.txt", "w", encoding="utf-8") as inverted:
        inverted.write("Term | Soundex | Appearances {Hash, [Frequency, Topic]}\n")

        # For each term, write the information down,, ig
        for term in index:
            inverted.write(f"{term} | {index[term][0]} | {json.dumps(index[term][1])}\n")
    # json.dump(index, open("index.json", "w", encoding="utf-8"), indent=4, sort_keys=True)
    return

def readInvertedIndex():
    # read the inverted index from the file
    index = {}
    with open("invertedindex.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:
        term, soundex, appearances = line.split("|")
        term = term.strip()
        soundex = soundex.strip()
        appearances = appearances.strip()
        documents = json.loads(appearances)
        index[term] = [soundex, documents]
    return index

def getLevenshteinDistance(suggestion, error):
    # possible operations: insertion, deletion, substitution (transposition is not considered)
    # insertion: add a letter (e.g. "cat" -> "cats" thus difference in length)
    # deletion: remove a letter (e.g. "cats" -> "cat" thus difference in length)
    # substitution: replace a letter (e.g. "cat" -> "cut" iterates through both words and finds each difference)

    distance = abs(len(suggestion)-len(error)) # initialize with difference in length
    # find the minimum length of the two words
    if len(suggestion) < len(error): 
        minlength = len(suggestion)
    else:
        minlength = len(error)
    # iterate through both words until we reach the end of one, for each difference found add 1 to the distance
    for i in range(minlength):
        if suggestion[i] != error[i]:
            distance += 1
    return distance # return the distance

def findBestWord(word, index):
    # find all terms that have the same soundex code as the word
    soundex = fuzzy.Soundex(4)
    wordSoundex = soundex(word)
    matches = []
    for term in index:
        if(index[term][0] == wordSoundex):
            matches.append(term)
    # get lowest levenshtein distance between the word and the matches
    if(len(matches) == 0): # if no matches are found, find lowest levenshtein distance between the word and all terms
        matches = list(index.keys())
    bestWord = matches[0]
    minimumDistance = getLevenshteinDistance(matches[0], word)
    for i in range(1, len(matches)):
        distance = getLevenshteinDistance(matches[i], word)
        if(distance < minimumDistance):
            bestWord = matches[i]
            minimumDistance = distance
        
    return bestWord

def searchForQuery():
    # user given prompt to enter query
    query = input("Enter your query: ").lower()
    terms = query.split(" ")
    # read the inverted index from the file
    try:
        index = readInvertedIndex()
    except FileNotFoundError:
        print("Index not found, try Index documents first.")
        return
    # json.dump(index, open("index.json", "w", encoding="utf-8"), indent=4, sort_keys=True)
    # index = json.load(open("index.json", "r", encoding="utf-8"))
    # For each word, if the word is not within inverted index terms, replace wrong word the best word using Soundex code. Now, you have a query that all words are available in the index.
    for i in range(len(terms)):
        if(terms[i] not in index):
            print(f"Term {printColors.BOLD}{terms[i]}{printColors.ENDC} not found in index, finding replacement...")
            # find the best word using soundex
            best = findBestWord(terms[i], index)
            print(f"Replacing {printColors.BOLD}{terms[i]}{printColors.ENDC} with {printColors.BOLD}{best}{printColors.ENDC}")
            terms[i] = best
    # find the all documents that contain the query terms
    documents = [(document, index[term][1][document][1]) for term in terms for document in index[term][1]]
    # vectorize documents and query using tfidf
    query = " ".join(terms)
    print(f"\nSearching for query: {printColors.BOLD}{query}{printColors.ENDC}")
    tfidfVec = TfidfVectorizer(lowercase=True, ngram_range = (1,1))
    similarities = []
    for document in documents:
        with open(f"data/{document[1]}/{document[0]}.txt", "r", encoding="utf-8") as f:
            text = f.read()
        vecDoc = tfidfVec.fit_transform([text]).toarray()
        vecQuery = tfidfVec.transform([query]).toarray()
        # calculate the cosine similarity of the documents
        similarities.append(cosine_similarity(vecDoc, vecQuery)[0][0])

    # print the top 3 most related documents
    print(f"\n{printColors.BOLD}Top 3 most related documents:{printColors.ENDC}")
    for i in range(3):
        if(len(similarities) == 0):
            print(f"{printColors.RED}There are no more documents that are similar to the query.{printColors.ENDC}")
            break
        maxIndex = similarities.index(max(similarities))
        # highlight query terms in your output using another color.
        with open(f"data/{documents[maxIndex][1]}/{documents[maxIndex][0]}.txt", "r", encoding="utf-8") as f:
            text = f.read()
        for term in terms:
            words = text.split(" ")
            words = [word.lower() for word in words]
            for j in range(len(words)):
                if(words[j] == term):
                    words[j] = "\033[1;31;40m" + words[j] + "\033[0m"
            text = " ".join(words)
        print(f"\n{printColors.BOLD}#{i+1}: Document <{documents[maxIndex][0]}.txt, {documents[maxIndex][1]}> : {round(similarities[maxIndex]*100, 2)}% Cosine similarity.{printColors.ENDC}\n")
        print(text)
        similarities.remove(similarities[maxIndex])
    print()
    return

def readDataset():
    # read the dataset from the file
    dataset = pd.DataFrame(columns=['text', 'topic', 'label', 'tokens', 'token-count'])
    datasetTokens = []
    for directory in os.listdir("data"):
        print("Reading directory: ", directory, "...")
        for filename in os.listdir(f"data/{directory}"):
            entry = {}
            if(filename[0] == '.'):
                continue
            with open(f"data/{directory}/{filename}", 'r', encoding="utf-8") as f:
                for line in f:
                    if(line == "\n"):
                        continue
                    entry['text'] = line
                    entry['topic'] = filename
                    if(directory == "Entertainment"):
                        entry['label'] = ENTERTAINMENT
                    elif(directory == "Lifestyle"):
                        entry['label'] = LIFESTYLE
                    elif(directory == "Technology"):
                        entry['label'] = TECHNOLOGY
                    entry["tokens"] = word_tokenize(entry['text'])
                    entry['token-count'] = len(entry['tokens'])
                    dataset = pd.concat([dataset, pd.DataFrame([entry])], ignore_index=True) # add the entry to the dataset dataframe
                    datasetTokens.append(' '.join(entry['tokens'])) # add the tokens to the datasetTokens list
    return dataset, datasetTokens

def bagOfWords(dataset):
    # create the bag of words vectorizer and dataset and save the vectorizer for use later
    cvr = CountVectorizer(max_features = 3000, lowercase=True, ngram_range = (1,1))
    bagOfWordsDataset = cvr.fit_transform(dataset).toarray()
    pickle.dump(cvr, open("ML-Models/vectorizer.model", 'wb'))
    return bagOfWordsDataset

def tfidf(dataset):
    # create the tfidf vectorizer and dataset and save the vectorizer for use later
    tfidfVec = TfidfVectorizer(max_features = 3000, lowercase=True, ngram_range = (1,1))
    tfidfDataset = tfidfVec.fit_transform(dataset).toarray()
    pickle.dump(tfidfVec, open("ML-Models/vectorizer.model", 'wb'))
    return tfidfDataset

def trainMLClassifier(dataset, datasetTokens, model, classifier):
    # train a classifier to predict a link's topic
    # vecorize the text using TF-IDF / bag of words
    if(model == "bag-of-words"): # if using bag of words model
        print("Using bag of words model with " + classifier + " classifier")
        bagOfWordsDataset = bagOfWords(datasetTokens)
        X_train, X_test, y_train, y_test = train_test_split(bagOfWordsDataset, dataset['label'].astype('int'), test_size=0.2, random_state=0)

    else: # if using tfidf model
        print("Using tfidf model with " + classifier + " classifier")
        tfidfDataset = tfidf(datasetTokens)
        X_train, X_test, y_train, y_test = train_test_split(tfidfDataset, dataset['label'].astype('int'), test_size=0.2, random_state=0)
    
    # create the classifier
    if(classifier == "naive-bayes"):
        clf = MultinomialNB()
    elif(classifier == "svm"):
        clf = SVC(kernel='linear', probability=True)
    elif(classifier == "decision-tree"):
        clf = DecisionTreeClassifier()
    else:
        # if no other classifier is specified, we must be using k-nearest neighbors
        n = int(''.join(c for c in classifier if c.isdigit())) # get the value of k from the argument by removing all non-digit characters
        clf = KNeighborsClassifier(n_neighbors=n)

    # train the model
    clf.fit(X_train, y_train)
    # predict the test data and find the accuracy of the model
    print("Testing model on test data.")
    predicted = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("Accuracy: " + str(accuracy))
    recall = metrics.recall_score(y_test, predicted, average="macro")
    print("Recall: " + str(recall))
    precision = metrics.precision_score(y_test, predicted, average="macro")
    print("Precision: " + str(precision))
    fScore = metrics.f1_score(y_test, predicted, average="macro")
    print("F1 Score: " + str(fScore))

    # plot the confusion matrix
    confusionMatrix = metrics.confusion_matrix(y_test, predicted)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=clf.classes_)
    disp.plot()
    plt.show()

    # dump the model to a file
    pickle.dump(clf, open("ML-Models/classifier.model", 'wb'))

    return accuracy, recall, precision, fScore

def predictLink(URL):
    try:
        page = requests.get(URL)
    except Exception as e:
        print("Error: Could not scrape URL: ", URL, " with error: ", e)
        return
    if(page.status_code != 200):
        print("Error: Could not scrape URL: ", URL, " with status code: ", page.status_code)
        return
    try:
        soup = BeautifulSoup(bytes(page.text, encoding="utf-8"), "html.parser")
    except Exception as e: # if the page is not in HTML format try to parse it as XML
        print(e)
        print("Could not parse page, returning...")
        return
    # get the text from the page
    content = getContentFromSoup(soup)

    # get the vectorizer and classifier
    try:
        with open("ML-Models/vectorizer.model", 'rb') as f:
            vectorizer = pickle.load(f)
        with open("ML-Models/classifier.model", 'rb') as f:
            classifier = pickle.load(f)
    except FileNotFoundError:
        print("Error: Could not find vectorizer or classifier model files. Please run Train ML classifier first.")
        return

    # vectorize the text
    vectorizedContent = vectorizer.transform([content]).toarray()

    # predict the topic of the link
    prediction = classifier.predict(vectorizedContent)
    confidence = classifier.predict_proba(vectorizedContent)
    if(prediction == ENTERTAINMENT):
        topic = "Entertainment"
    elif(prediction == LIFESTYLE):
        topic = "Lifestyle"
    elif(prediction == TECHNOLOGY):
        topic = "Technology"
    if(confidence[0][prediction[0]] < 0.5):
        print(f"{printColors.RED}Predicted topic: <{topic}, {round((1-confidence[0][prediction[0]])*100, 2)}%>{printColors.ENDC}")
    elif(confidence[0][prediction[0]] > 0.9):
        print(f"{printColors.GREEN}Predicted topic: <{topic}, {round(confidence[0][prediction[0]]*100, 2)}%>{printColors.ENDC}")
    else:
        print(f"{printColors.YELLOW}Predicted topic: <{topic}, {round(confidence[0][prediction[0]]*100, 2)}%>{printColors.ENDC}")


    return prediction[0]

if __name__ == "__main__":
    # get input from user
    while True:
        print("Select an option: ")
        print()
        print("\t1 - Collect new documents.")
        print("\t2 - Index documents.")
        print("\t3 - Search for a query.")
        print("\t4 - Train ML classifier.")
        print("\t5 - Predict a link.")
        print("\t6 - Quit.")
        print()
        
        num = input("Enter your choice: ")
        
        if(num == '1'):
            print("Collecting new documents...")
            collectDocuments()
        elif(num == '2'):
            print("Indexing documents...")
            indexDocuments()
        elif(num == '3'):
            searchForQuery()
        elif(num == '4'):
            vec = "tfidf"
            clf = "svm"
            print("Training ML classifier with " + vec + " vectorizer and " + clf + " classifier.")
            print("Reading dataset...")
            dataset, datasetTokens = readDataset()
            if(dataset.empty):
                print("Error: Dataset is empty. Please run Collect new documents first.")
                continue
            print("Training ML classifier...")
            scores = trainMLClassifier(dataset, datasetTokens, vec, clf)
        elif(num == '5'):
            link = input("Enter the link to predict: ")
            print(f"Predicting topic of {link}...")
            predictLink(link)
        elif(num == '6'):
            print("Quitting...")
            break
        else:
            print("\nInvalid choice. Try again.\n")
    