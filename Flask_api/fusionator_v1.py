import spacy
from joblib import load

splog_dr = load(filename="splog_dr.pkl")
spada_hr = load(filename="spada_hr.pkl")
spgrad_ar = load(filename="spgrad_ar.pkl")
sprand_rs = load(filename="sprand_rs.pkl")
spada_rt = load(filename="spada_rt.pkl")

nlp = spacy.load("en_core_web_lg")

# Function for preprocessing
def preprocess(text, spacy_model=nlp):
    """Pass a text and it will preprocess it!"""
    filtered = []
    doc = spacy_model(text)
    for token in doc:
        if (not token.is_stop) and (not token.is_punct):
            filtered.append(token.lemma_)

    filt_txt = " ".join(filtered)
    return filt_txt

def fusionator_v1(text, models=[splog_dr, spada_hr, spgrad_ar, sprand_rs, spada_rt],
                  nlp=nlp):
    ptext = preprocess(text)
    doc = nlp(ptext)
    doc_vec = doc.vector

    i = 0
    rom_prob = 0
    dra_prob = 0
    hor_prob = 0
    act_prob = 0
    sci_prob = 0
    thr_prob = 0

    while i < len(models):
        if i <= 2:
            prob = models[i].predict_proba([doc_vec])
            rom_prob += prob[0][1]
            if models[i].predict([doc_vec]) == "drama":
                dra_prob += prob[0][0]
            if models[i].predict([doc_vec]) == "horror":
                hor_prob += prob[0][0]
            if models[i].predict([doc_vec]) == "action":
                act_prob += prob[0][0]

        elif i > 2 and i <= 5:
            prob = models[i].predict_proba([doc_vec])
            # Reverse here
            # First one belong to label r
            rom_prob += prob[0][0]
            if models[i].predict([doc_vec]) == "scifi":
                sci_prob += prob[0][1]
            if models[i].predict([doc_vec]) == "thriller":
                thr_prob += prob[0][1]

        i += 1
        # Loop ends here
    rom_prob = rom_prob / 5
    sum_prob = rom_prob + dra_prob + hor_prob + act_prob + sci_prob + thr_prob

    rom_prob_f = (rom_prob / sum_prob) * 100
    dra_prob_f = (dra_prob / sum_prob) * 100
    hor_prob_f = (hor_prob / sum_prob) * 100
    act_prob_f = (act_prob / sum_prob) * 100
    sci_prob_f = (sci_prob / sum_prob) * 100
    thr_prob_f = (thr_prob / sum_prob) * 100

    output = {
        "romantic":rom_prob_f,
        "drama":dra_prob_f,
        "horror":hor_prob_f,
        "action":act_prob_f,
        "sci-fi":sci_prob_f,
        "thriller":thr_prob_f
    }

    return output

if __name__ == "__main__":
    user_input = input("Copy paste a synopsis here:\n")
    res = fusionator_v1(user_input)
    print(f"\n{res}")
