# This script is just for saving the data, content based filtering

import pandas as pd
import spacy
import pickle

nlp = spacy.load("en_core_web_lg")
df = pd.read_json("../data_json/imbd_json_data.json", lines=True)

def preprocess(text, spacy_model=nlp):
    filtered = []
    doc = spacy_model(text)
    for token in doc:
        if (not token.is_stop) and (not token.is_punct):
            filtered.append(token.lemma_)

    return " ".join(filtered)

def vecword(text, nlp=nlp):
    doc = nlp(text)
    doc_vec = doc.vector
    return doc_vec

def convert_vec(syn):
    ptext = preprocess(text=syn)
    ptext_vec = vecword(text=ptext)
    return ptext_vec

df["vec"] = df["Overview"].apply(convert_vec)
with open("movie_data_vec.pkl", "wb") as f:
    pickle.dump(obj=df, file=f)
print("Data with vector folder has been saved with word vectors!")


# print(df[["Overview", "vec"]].sample(10))

