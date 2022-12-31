'''
spam_filter.py
Spam v. Ham Classifier trained and deployable upon short
phone text messages.
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *

class SpamFilter:

    def __init__(self, text_train, labels_train):
        """
        Creates a new text-message SpamFilter trained on the given text 
        messages and their associated labels. Performs any necessary
        preprocessing before training the SpamFilter's Naive Bayes Classifier.
        As part of this process, trains and stores the CountVectorizer used
        in the feature extraction process.
        
        :param DataFrame text_train: Pandas DataFrame consisting of the
        sample rows of text messages
        :param DataFrame labels_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each text message
        """
        vectorizer = CountVectorizer(stop_words= "english")
        vectorizer.fit(text_train) 
        features = vectorizer.transform(text_train) 
        print("vectorized vocab:  " + str(vectorizer.vocabulary_))
        self.vectorizer = vectorizer

        learned_nbc = MultinomialNB()
        learned_nbc.fit(features, labels_train) 
        self.learned_nbc = learned_nbc

        
        
    def classify (self, text_test):
        """
        Takes as input a list of raw text-messages, uses the SpamFilter's
        vectorizer to convert these into the known bag of words, and then
        returns a list of classifications, one for each input text
        
        :param list/DataFrame text_test: A list of text-messages (strings) consisting
        of the messages the SpamFilter must classify as spam or ham
        :return: A list of classifications, one for each input text message
        where index in the output classes corresponds to index of the input text.

        """
        features = self.vectorizer.transform(text_test) #input text messages into their numerical format
        predicted_labels = self.learned_nbc.predict(features) #save predicted label array
        return predicted_labels

    
    def test_model (self, text_test, labels_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test texts
        and their associated labels), classifies each text, and then prints
        the classification_report on the expected vs. given labels.
        
        :param DataFrame text_test: Pandas DataFrame consisting of the
        test rows of text messages
        :param DataFrame labels_test: Pandas DataFrame consisting of the
        test rows of labels pertaining to each text message
        
        """
        classified = self.classify(text_test)
        print(str(classification_report(labels_test, classified)))

    
        
def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with only the message
    texts and labels as the remaining columns.
    
    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the texts
    and labels

    """
    data = pd.read_csv(data_file, encoding="latin-1")
    data.drop(data.columns[[2,3,4]], axis = 1, inplace = True)

    print("columns " + str(data.columns))

    data.rename(columns={"v1": "class", "v2": "text"}, inplace=True)
    print(data.head())
    return data


if __name__ == "__main__":
    """ Train / Test Data Split """
   
    data = load_and_sanitize("../dat/texts.csv")
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["class"])
    datay = SpamFilter(text_train= X_train,labels_train= y_train)

    ham_text1 = ["Hey sweetie. Just wanted to make sure the diamond parking for $15.70 and the New York Times charge for $58.89 were done by you! "]
    ham_text2 = ["Don’t stress out if your flight is delayed. It is the busiest travel day of the year and we will just deal with whatever time you get here, OK!"]
    spam_text1 = ["I love DRUGZ and BUTTZ. tbh im not evn sure what spam is ???? like who cars"]
    spam_text2 = ["FREE DRUGS!!! i luv drugs. drugs R cool. Please male me some drugs"]

    print("ham test:  " + str(datay.classify(ham_text1)))
    print("ham test2:  " + str(datay.classify(ham_text2)))
    print("spam test:  " + str(datay.classify(spam_text1)))
    print("spam test2:  " + str(datay.classify(spam_text2)))

    print("-- testing the model w/ X_test:  " + str(datay.classify(X_test)))
    print("-- classification report w/ Y_test: " )
    print(str(classification_report(y_test, datay.classify(X_test))))

    print("+++++++++++++++++ Testing the Model+++++++++++++++++")
    datay.test_model(X_test, y_test)

   