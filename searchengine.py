
def collectDocuments():
    # crawl each link in sources.txt
    # extract text from each link using jusText, trafilatura
    # For Crawling, if the page is not available crawl it. For availability, you can check hash value of URL and its existence in related subfolder
    # remove stopwords
    # save content in a file to data/topic folder (name is H.txt, where H is the hash value of the URL)
    # update crawl.log file <topic, link’s URL, Hash value of URL, date>
    # only crawl pages that have the same root domain as the seed URL
    return
    
def indexDocuments():
    return
    
def searchForQuery():
    # user given prompt to enter query
    return

def trainMLClassifier():
    # train a classifier to predict a link's topic
    return

def predictLink():
    return

def yourStory():
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
        
        num = int(input("Enter your choice: "))
        
        if(num == 1):
            print("Collecting new documents...")
            collectDocuments()
        elif(num == 2):
            print("Indexing documents...")
            indexDocuments()
        elif(num == 3):
            print("Searching for a query...")
            searchForQuery()
        elif(num == 4):
            print("Training ML classifier...")
            trainMLClassifier()
        elif(num == 5):
            print("Predicting a link...")
            predictLink()
        elif(num == 6):
            print("Your story...")
            yourStory()
        elif(num == 7):
            print("Quitting...")
            break
        else:
            print("Invalid choice. Try again.")
    