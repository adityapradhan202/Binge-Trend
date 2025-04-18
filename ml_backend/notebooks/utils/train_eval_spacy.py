# This script contains a modified version of train_eval.py
# But functions of this script are in the context of spacy word vectors

import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

# Sklearn algos for testing only
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Large english model for word vectors
nlp_large = spacy.load("en_core_web_lg")

def preprocess(text, spacy_model=nlp_large):
    """Pass a text and it will preprocess it!
    Args:
        text: text that has to be preprocessed.
        spacy_model: language model of the spacy library, small english model is recommended.
    Returns:
        processed_text: rerturns processed text, which doesn't have any stop words or punctuation marks.
    """

    filtered = []
    doc = spacy_model(text)
    for token in doc:
        if (not token.is_stop) and (not token.is_punct):
            filtered.append(token.text)

    return " ".join(filtered)


def vecword(text, nlp=nlp_large):
    doc = nlp(text)
    doc_vec = doc.vector
    return doc_vec

def spacy_vector_train(models, df, X, y, nlp=nlp_large, 
                       normalize=True,tsize=0.20,rstate=45, cfreport=True, max_performer=True, complete_res=True):
    eval_res = {}

    print("\nWait please...")
    print("Preprocessing the text...")
    df[X] = df[X].apply(preprocess)
    print("Wait please...")
    print("Converting the text into spacy word vectors...")
    df["vec"] = df[X].apply(vecword)
    print("-> Text has been converted into word vectors!")

    X_train, X_test, y_train, y_test = train_test_split(
        df["vec"], df[y], test_size=tsize, random_state=rstate)
    
    X_train_2d = np.stack(arrays=X_train, axis=0)
    X_test_2d = np.stack(arrays=X_test, axis=0)

    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(X_train_2d)
        X_train_f = scaler.transform(X_train_2d)
        X_test_f = scaler.transform(X_test_2d)
    else:
        X_train_f, X_test_f = X_train_2d, X_test_2d

    print("\n-> Training the models now:")
    for model in models:
        model.fit(X_train_f, y_train)
        preds = model.predict(X_test_f)
        acc = accuracy_score(y_test, preds)
        print(f"Model name: {model} | Acc: {acc}")

        if cfreport:
            report = classification_report(y_test, preds)
            print(f"Classification report:\n{report}")
            eval_res[str(model)] = round(acc * 100, 2)

    if complete_res:
        print("Complete results for all algorithns:")
        print(eval_res)
        
    if max_performer:
        max_a = 0
        max_p = ""
        for model in eval_res:
            if eval_res[model] > max_a:
                max_a = eval_res[model]
                max_p = model
        return max_p, max_a
        
if __name__ == "__main__":

    df = pd.read_excel("D:/projects-aiml/Binge-Trend/ml_backend/data_sets/rotten_tomatoes_100.xlsx")
    max_p, max_a = spacy_vector_train(
        models=[
            MultinomialNB(),
            AdaBoostClassifier(algorithm="SAMME"),
            GradientBoostingClassifier()
        ],
        df=df, X="synopsis", y="label",
        nlp=nlp_large,
        normalize=True,
        tsize=0.20,
        rstate=45,
        cfreport=True,
        max_performer=True,
        complete_res=False
    )
    







