# Util functions, for training and evaluating algorithms.

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import spacy

nlp = spacy.load("en_core_web_sm")
def preprocess(text, spacy_model=nlp):
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

def base_train_eval(models, X, y, tsize=0.20, rstate=45, 
                    vec_type="tfidf", acc=True, cfreport=True, show_vocab=True, max_performer=True,
                    complete_res=True):
    
    """
    Applies all the base form of algorithms to the dataset directly, and performs text classification on all the classes that are present in the entire dataset.
    Args:
        models: list of base models or algorithms.
        X: X label, description or synopsis here.
        y: y label, genre here.
        tsize: test size for train test split.
        rstate: random state int value for train test split.
        vec_type: type of vectorizer, set either "tfidf", or "count" for TfIdf or CountVectorizer respectively.
        acc: default value is True, if true then it calculates accuracy score of each algorithm.
        cfreport: default value is True, if true then it calculates the classification report for each algorithm.
        show_vocab: default value is True, if true then it provides a glimpse of the bag of words.
        max_performer: default value is True, if true then it displays the best performing algorithm.
        complete_res: default value is True, if true then it displays the results of all the algorithms.
    Returns:
        max_p: max performer, or the best performing algorithm.
        max_a: accuracy score of the max_p.
    """
    
    eval_res = {} # Results of evaluation process
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, random_state=rstate)
    if vec_type == "tfidf":
        vec = TfidfVectorizer()
    elif vec_type == "count":
        vec = CountVectorizer()
    else:
        print("Enter a valid string for choosing the vectorizer!")
        return # terminates here
    
    vec.fit(X_train)
    spX_train = vec.transform(X_train)
    spX_test = vec.transform(X_test)

    if show_vocab == True:
        # bow = vec.vocabulary_ (each word is mapped to an index)
        bow = vec.get_feature_names_out()[200:300] # Just a glimpse of BOW
        print("\n---> Given below is the BOW:\n")
        print(bow)
        print()
    
    print("\n----> Classification report of the classification algorithms")
    for model in models:
        model.fit(spX_train, y_train)
        preds = model.predict(spX_test)

        print(f"Model Name: {model}")
        if acc == True:
            acc_score = accuracy_score(y_true=y_test, y_pred=preds)
            print(f"Overall accuracy: {round(acc_score * 100, 2)} %")
        if cfreport == True:
            report = classification_report(y_true=y_test, y_pred=preds)
            print(f"Classification report:\n{report}")
        if max_performer == True:
            eval_res[str(model)] = round(acc_score * 100, 2)

    if complete_res == True:
        print("Complete results for all algorithms:")
        print(eval_res)

    if max_performer == True:
        max_a = 0 
        max_p = ""
        for model in eval_res:
            if eval_res[model] > max_a:
                max_a = eval_res[model]
                max_p = model
        # print(f"\n\n---> Max performer: {max_p}")
        # print(f"---> Overall accuracy of this model: {max_a}")

        return max_p, max_a # return max performer and it's accuracy

def pairs_train_eval(y, data, label_col, synop_col, models=None, 
                     tsize=0.20, rstate=45, vec_type="tfidf", acc=True,
                     cfreport=True, show_vocab=True, max_performer=True, complete_res=True,
                    ):
    """
    Trains, tests, evaluates, and shows result of algorithms on every possible pair of class.
    Args:
        y: label column, pandas series, for example df["syopsis"]
        data: pandas dataframe, of type pd.DataFrame
        synop_col: string of the synopsis column
        models: list of all the models and algorithms
        tsize: test size for train test split
        rstate: random state int value for train test split
        vec_type: type of vector, set either "tfidf" or "count" for TfIdfVectorizer or CountVecotrizer
        acc: default values is true, if set true then it calculates and displays the accuracy score
        cfreport: defaault value is truem if set true then it calculates and displays the classification report
        show_vocab: default value is true, if set true then it displays a glimpse of the bag of words
        max_performer: default value is true, if set true then it displays the best performing algorithm
        complete_res: default value is true, if set true then it display the result of all the algorithms
    Returns:
        final_res: a dictionary containing the final result, with the pair, best algorithm and it's accuracy score
    """
    label_pairs = set()
    all_labels = list(y.unique())
    for label in all_labels:
        label_current = label
        for label in all_labels:
            if label_current != label:
                pair = str(label_current) + "-" + str(label)
                label_pairs.add(pair)

    label_pairs = list(label_pairs)
    final_res = {} # Final result from the train & eval
    for i, lb in enumerate(label_pairs):
        lb1 = lb.split("-")[0]
        lb2 = lb.split("-")[1]
        filt_data = data[(data[label_col]==lb1) | (data[label_col]==lb2)] # filtered segment

        print(f"\n\n---> Pair-{i + 1}, Pair Name: {lb}\n\n")
        max_p, max_a = base_train_eval(
            models=models,
            X=filt_data[synop_col],
            y=filt_data[label_col],
            tsize=tsize,
            rstate=rstate,
            vec_type=vec_type,
            acc=acc,
            cfreport=cfreport,
            show_vocab=show_vocab,
            max_performer=max_performer,
            complete_res=complete_res
        )

        final_res[lb] = f"{max_p}  <---> {max_a:.2f}"
        # filt_data = data[(data[y]==lb1) | (data[y]==lb2)]
        # print(filt_data.sample(10)[])
        # print(lb1,lb2)

    return final_res

def train_fine_tune(models,):
    """Quickle trains and evaluates the base form of models, and also fine tunes them, if the fine tuned models are better than the base form of models, then it returns the fine tuned models, otherwise returns the base form of models."""
    pass


if __name__ == "__main__":
    pass