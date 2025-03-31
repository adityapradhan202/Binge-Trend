from joblib import load
import spacy

loaded_log_rd = load(filename="../models/log_rd.pickle")
loaded_nb = load(filename="../models/nb.pickle")
loaded_nb2 = load(filename="../models/nb2.pickle")
loaded_nb_rs = load(filename="../models/nb_rs.pickle")
loaded_log_rt = load(filename="../models/log_rt.pickle")

loaded_vec1 = load(filename="../vectorizers/vec1.pickle")
loaded_vec2 = load(filename="../vectorizers/vec2.pickle")
loaded_vec3 = load(filename="../vectorizers/vec3.pickle")
loaded_vec4 = load(filename="../vectorizers/vec4.pickle")
loaded_vec5 = load(filename="../vectorizers/vec5.pickle")

nlp = spacy.load("en_core_web_sm")
def preprocess(text, spacy_model=nlp):
    """Pass a text and it will preprocess it!"""
    filtered = []
    doc = spacy_model(text)
    for token in doc:
        if (not token.is_stop) and (not token.is_punct):
            filtered.append(token.text)

    return " ".join(filtered)

def fusionator_v0(text, models, vecs):
    """
    Just provide the text and it predict the genres!

    Args:
        text: synopsis of movie or web series or whatever you are watching
        models: list of all the saved models
        vecs: list of all the saved vectorizers

    Returns:
        None: Returns nothing, but displays the prediction probabilities of the genres.
    """
    
    ptext = preprocess(text)
    i = 0
    rom_prob = 0
    dra_prob = 0
    hor_prob = 0
    act_prob = 0
    sci_prob = 0
    thr_prob = 0

    while i < len(vecs):
        if i <= 2:
            ptext_vec = vecs[i].transform([ptext])
            prob = models[i].predict_proba(ptext_vec)
            rom_prob += prob[0][1]

            if models[i].predict(ptext_vec) == "drama":
                dra_prob += prob[0][0]
            if models[i].predict(ptext_vec) == "horror":
                hor_prob += prob[0][0]
            if models[i].predict(ptext_vec) == "action":
                act_prob += prob[0][0]

        elif i > 2 and i <= 5:
            ptext_vec = vecs[i].transform([ptext])
            prob = models[i].predict_proba(ptext_vec)

            # reverse here (First one belongs to label r)
            rom_prob += prob[0][0]
            if models[i].predict(ptext_vec) == "scifi":
                sci_prob += prob[0][1]
            if models[i].predict(ptext_vec) == "thriller":
                thr_prob += prob[0][1]
        i += 1 
        # loop ends here
    rom_prob = rom_prob / 5
    sum_prob = rom_prob + dra_prob + hor_prob + act_prob + sci_prob + thr_prob

    rum_prob_f = (rom_prob / sum_prob) * 100
    dra_prob_f = (dra_prob / sum_prob) * 100
    hor_prob_f = (hor_prob / sum_prob) * 100
    act_prob_f = (act_prob / sum_prob) * 100
    sci_prob_f = (sci_prob / sum_prob) * 100
    thr_prob_f = (thr_prob / sum_prob) * 100

    # Final results (combined)
    print(f"Romance: {rum_prob_f}")
    print(f"Drama: {dra_prob_f}")
    print(f"Horror: {hor_prob_f}")
    print(f"Action: {act_prob_f}")
    print(f"Scifi: {sci_prob_f}")
    print(f"Thriller: {thr_prob_f}")


if __name__ == "__main__":
    syn = input("Copy paste the synopsis here:\n")
    fusionator_v0(
        text=syn,
        models=[
            loaded_log_rd,
            loaded_nb,
            loaded_nb2,
            loaded_nb_rs,
            loaded_log_rt],
        vecs=[
            loaded_vec1, 
            loaded_vec2,
            loaded_vec3,
            loaded_vec4,
            loaded_vec5]
    )

    