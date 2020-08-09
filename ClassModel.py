# import libraries
import csv
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# dataset
text_file = 'ivr_data.csv'
data = "TEXTS"
labels = "LABELS"


class Classifier():
    def __init__(self):
        self.train_set, self.test_set = self.load_data()
        self.counts, self.test_counts = self.vectorize()
        self.classifier = self.train_model()

    def load_data(self):
        df = pd.read_csv(text_file, header=0, error_bad_lines=False)
        train_set, test_set = train_test_split(df, test_size=.15, random_state=1)
        return train_set, test_set

    def train_model(self):
        classifier = MultinomialNB(alpha=1)
        targets = self.train_set[labels]
        classifier.fit(self.counts, targets)
        return classifier

    def vectorize(self):
        vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True)
        counts = vectorizer.fit_transform(self.train_set[data])
        test_counts = vectorizer.transform(self.test_set[data])
        self._vectorizer = vectorizer
        return counts, test_counts

    def evaluate(self):
        test_counts,test_set = self.test_counts, self.test_set
        predictions = self.classifier.predict(test_counts)
        print (classification_report(test_set[labels], predictions))
        

    def classify(self, input):
        input_text = input
        input_counts = self._vectorizer.transform(input_text)
        predictions = self.classifier.predict(input_counts)
        return predictions
        
        
def main():
    myModel = Classifier()
    with open('modelpicklefinal', 'wb') as modelnb:
        pickle.dump(myModel, modelnb)
        modelnb.close()


if __name__=='__main__':
    main()

