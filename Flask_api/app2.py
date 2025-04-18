# This is the second api file
# This part is for matching movies
# Second api will by default use the first api

from flask import Flask, request, jsonify
from flask_cors import CORS

# For usage of first api
from fusionator_v0 import fusionator_v0

from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pandas as pd

# Spacy large model
nlp = spacy.load("en_core_web_lg")
df = pd.read_pickle("movie_data_vec.pkl")

def preprocess_vec(text, nlp=nlp):
    doc = nlp(text)
    filtered = []
    for token in doc:
        if (not token.is_stop) and (not token.is_punct):
            filtered.append(token.lemma_)

    ptext = " ".join(filtered)
    docvec = nlp(ptext).vector
    return docvec

def find_sim(syn, df=df, thresh=0.85):
    inds = []
    inp_vec = preprocess_vec(syn)
    while thresh > 0.50:
        for ind, vec in enumerate(df["vec"]):
            csim = cosine_similarity(X=[inp_vec], Y=[vec])[0][0]
            if csim > thresh:
                inds.append(ind)
        if len(inds) != 0:
            break
        else:
            thresh -= 0.05 # Decrement threshold by 0.05
    # while loop ends here
            
    res = {}
    for i, ind in enumerate(inds):
        genres = fusionator_v0(text=df.iloc[ind]["Overview"])
        res[str(i)] = {
            "title":df.iloc[ind]["Series_Title"],
            "poster_url":df.iloc[ind]["Poster_Link"],
            "released_year":df.iloc[ind]["Released_Year"],
            "runtime":df.iloc[ind]["Runtime"],
            "imdb_rating":df.iloc[ind]["IMDB_Rating"],
            "genres":genres
        }

    return res


app = Flask(__name__)
CORS(app)
@app.route('/find-match', methods=['POST'])
def predict():
    data = request.get_json()
    synopsis = data.get('synopsis', '')
    result = find_sim(syn=synopsis)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=6000)