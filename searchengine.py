# Import libraries
import os.path
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from hashlib import blake2b
import justext
from nltk.corpus import stopwords
import warnings

# nltk libraries
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Sklearn training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split

# Sklearn Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Sklearn metrics
from sklearn import metrics

# pickle for saving the model
import pickle

# visualization
import matplotlib.pyplot as plt

MAXDEPTH = 2

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
    with open("crawler1.log", "a") as f:
        # <topic, link’s URL, Hash value of URL, date>
        f.write(f"<{topic}, {URL}, {H}, {datetime.now()}, {page.status_code}>\n")
    
    if(page.status_code != 200):
        # print("Error: Could not scrape URL: ", URL)
        return

    # print("Scraping URL: ", URL)
    # Initialize the BeautifulSoup object
    # Make warnings into errors so that we can catch them
    warnings.filterwarnings("error")
    try:
        soup = BeautifulSoup(bytes(page.text, encoding="utf-8"), "html.parser")
    except Exception as e: # if the page is not in HTML format try to parse it as XML
        print(e)
        print("Could not parse page as HTML. Trying XML.")
        soup = BeautifulSoup(bytes(page.text, encoding="utf-8"), "xml")
    # Reset the warning behavior
    warnings.resetwarnings()

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
            
def readSources():
    # read the sources.txt file and return the list of links
    with open("sources.txt", "r") as f:
        links = f.readlines()
    return links

def collectDocuments():
    # crawl each link in sources.txt
    if(not os.path.isfile("sources.txt")):
        raise("sources.txt not found.")
    
    links = readSources()

    for link in links:
        if(link[0] == "#"):
            # This is a comment
            continue
        topic, URL = link.split(",")
        URL = URL.strip()
        topic = topic.strip()
        print("Scraping URL: ", URL)
        scrapeURL(URL, MAXDEPTH, topic)
    # For Crawling, if the page is not available crawl it. For availability, you can check hash value of URL and its existence in related subfolder
    # update crawl.log file <topic, link’s URL, Hash value of URL, date>
    # only crawl pages that have the same root domain as the seed URL
    return

def findEmptyFiles():
    empty = 0
    total = 0
    for directory in os.listdir("data"):
        for file in os.listdir(f"data/{directory}"):
            with open(f"data/{directory}/{file}", "r", encoding="utf-8") as f:
                text = f.read()
                if(len(text) == 0):
                    print(f"data/{directory}/{file}")
                    empty += 1
                total += 1
    print(empty / total * 100, "% of the documents are empty.")
    return

def indexDocuments():
    findEmptyFiles()
    return
    
def searchForQuery():
    # user given prompt to enter query
    return

def readDataset():
    # read the dataset from the file
    dataset = pd.DataFrame(columns=['text', 'topic', 'label', 'tokens', 'token-count'])
    datasetTokens = []
    for directory in os.listdir("data"):
        print("Reading directory: ", directory, "...")
        for filename in os.listdir(f"data/{directory}"):
            entry = {}
            with open(f"data/{directory}/{filename}", 'r', encoding="utf-8") as f:
                for line in f:
                    if(line == "\n"):
                        continue
                    entry['text'] = line
                    entry['topic'] = filename
                    if(directory == "Entertainment"):
                        entry['label'] = 0
                    elif(directory == "Lifestyle"):
                        entry['label'] = 1
                    elif(directory == "Technology"):
                        entry['label'] = 2
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
        clf = SVC(kernel='linear')
    elif(classifier == "decision-tree"):
        clf = DecisionTreeClassifier()
    else:
        # if no other classifier is specified, we must be using k-nearest neighbors
        n = int(''.join(c for c in classifier if c.isdigit())) # get the value of k from the argument by removing all non-digit characters
        print("Using k-nearest neighbors with k = " + str(n))
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
    # user given prompt to enter a link
    # scape the link's text using jusText, trafilatura
    # predict the topic of the link content using the trained classifier
    try:
        page = requests.get(URL)
    except Exception as e:
        print("Error: Could not scrape URL: ", URL, " with error: ", e)
        return
    if(page.status_code != 200):
        print("Error: Could not scrape URL: ", URL, " with status code: ", page.status_code)
        return
    # Make warnings into errors so that we can catch them
    warnings.filterwarnings("error")
    try:
        soup = BeautifulSoup(bytes(page.text, encoding="utf-8"), "html.parser")
    except Exception as e: # if the page is not in HTML format try to parse it as XML
        print(e)
        print("Could not parse page as HTML. Trying XML.")
        soup = BeautifulSoup(bytes(page.text, encoding="utf-8"), "xml")
    # Reset the warning behavior
    warnings.resetwarnings()
    # get the text from the page
    content = getContentFromSoup(soup)

    # get the vectorizer and classifier
    with open("ML-Models/vectorizer.model", 'rb') as f:
        vectorizer = pickle.load(f)
    with open("ML-Models/classifier.model", 'rb') as f:
        classifier = pickle.load(f)

    # vectorize the text
    vectorizedContent = vectorizer.transform([content]).toarray()

    # predict the topic of the link
    prediction = classifier.predict(vectorizedContent)
    confidence = classifier.predict_proba(vectorizedContent)
    if(prediction == 0):
        topic = "Entertainment"
    elif(prediction == 1):
        topic = "Lifestyle"
    elif(prediction == 2):
        topic = "Technology"
    print(f"Predicted topic: <{topic}, %{round(confidence[0][prediction[0]]*100, 2)}>")
    return prediction[0]

def yourStory():
    # read the story from story.txt
    return

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
        print("\t6 - Your story!")
        print("\t7 - Quit.")
        print()
        
        num = input("Enter your choice: ")
        
        if(num == '1'):
            print("Collecting new documents...")
            collectDocuments()
        elif(num == '2'):
            print("Indexing documents...")
            indexDocuments()
        elif(num == '3'):
            print("Searching for a query...")
            searchForQuery()
        elif(num == '4'):
            print("Training ML classifier...")
            print("Enter the vectorizer to use: ")
            print("\t1 - Bag of words.")
            print("\t2 - TF-IDF.")
            print()
            vec = input("Enter your choice: ")

            print("Enter the classifier to use: ")
            print("\t1 - Naive Bayes.")
            print("\t2 - SVM.")
            print("\t3 - Decision Tree.")
            print("\t4 - K-Nearest Neighbors.")
            print()
            clf = input("Enter your choice: ")

            if(vec == '1'):
                vec = "bag-of-words"
            else:
                vec = "tfidf"
            
            if(clf == '1'):
                clf = "naive-bayes"
            elif(clf == '2'):
                clf = "svm"
            elif(clf == '3'):
                clf = "decision-tree"
            else:
                print("Enter the value of k for k-nearest neighbors: ")
                k = input("Enter your choice: ")
                clf = "k-nearest-neighbors-" + k
            print("Training ML classifier with " + vec + " vectorizer and " + clf + " classifier.")
            print("Reading dataset...")
            dataset, datasetTokens = readDataset()
            print("Training ML classifier...")
            scores = trainMLClassifier(dataset, datasetTokens, vec, clf)
            print("dumping dataset to csv.")
            dataset.to_csv("ML-Scores/dataset.csv", index=False)
            print("Dumping model scores to csv.")
            if(os.path.exists("ML-Scores/scores.csv")):
                df = pd.read_csv("scores.csv")
            else:
                df = pd.DataFrame(columns=["classifier", "model", "accuracy", "recall", "precision", "f1"])
            # concat the new scores to the dataframe
            df = pd.concat([df, pd.DataFrame([[clf, vec, scores[0], scores[1], scores[2], scores[3]]], columns=["classifier", "model", "accuracy", "recall", "precision", "f1"])])
            # write the dataframe to the csv
            df.to_csv("ML-Scores/scores.csv", index=False)
        elif(num == '5'):
            link = input("Enter the link to predict: ")
            print(f"Predicting topic of {link}...")
            predictLink(link)
        elif(num == '6'):
            print("Your story...")
            yourStory()
        elif(num == '7'):
            print("Quitting...")
            break
        else:
            print("\nInvalid choice. Try again.\n")
    