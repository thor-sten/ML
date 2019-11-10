"""
Support ticket classifier based on FastText

Thorsten Jacobs

https://github.com/karolzak/support-tickets-classification
"""

import data
import re
# import random
# from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import fastText
from fastText import train_supervised

# Data preprocessing

path = "data\\"

def simplify(string):
    """Input: String. Output: Simplified sting (lowercase, no special characters, ...)."""
    string = string.lower()
    string = re.sub("[(\xa0)(\n)]", " ", string) # replace non-breaking space and newline
    string = re.sub("\W+", " ", string) # replace any non-alphanumeric character
    string = re.sub("\d+", "0", string) # replace all digits with 0
    string = re.sub(" +", " ", string) # replace multiple spaces 
    return string


def createLabels(classes, text):
    """Input: List of classes, text string. Output: Labelled text lines for fasttext."""
    labels = ''
    if len(classes) > 1:
        for cl in classes:
            labels += "__label__{} ".format(cl)
    elif len(classes) == 1:
        labels = "__label__{} ".format(classes.pop())

    return labels + text


# Classify

def classify(describtionText, numberOfClasses=10, printResult=True):
    """Input: String with product description text. Output: Suggested Co classes"""
    preprocessed = tmData.simplify(describtionText)
    labels, probability = classifier.predict(preprocessed, numberOfClasses)
    labels = [cc[9:] for cc in labels] # remove __label__ text
    probability = (np.around(probability, decimals=4))

    print(preprocessed)
    if printResult:
        print('\nClass \t Score')
        for i, lab in enumerate(labels):
            print(lab, '\t', probability[i])
    
    return labels


# Save and load model to/from file
SAVEMODEL = False
LOADMODEL = False

if LOADMODEL:
    classifier = fastText.load_model(path + "model.bin")

trainfile = "PRKO_train.txt"  # Components, PR mapping
testfile = "PRKO_test.txt"

if not LOADMODEL:
    # Build and test model using fastText. Options: https://fasttext.cc/docs/en/options.html
    classifier = train_supervised(input=path + trainfile, epoch=25, lr=0.5)
    result = classifier.test(path + testfile)

    try:
        F1 = 2 * result[1] * result[2] / (result[1] + result[2])
    except:
        print("Error: No result available.")
    else:
        print("Test samples \t Precision \t Recall \t F1 \n", result, F1)

    if SAVEMODEL:
        classifier.save_model(path + "model.bin")


# User promt
numberOfCl = 10
while True:
    descr = input("\nEnter product description (q to quit): ")
    if descr == 'quit' or descr == 'q':
        print('Program ended\n')
        break

    classify(descr, numberOfCl)
