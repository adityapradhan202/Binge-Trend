# Function for training specific algorithms
# Also fine tuning them and comparing to the base form of algorithms...

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

class FineTuneError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Train and Fine tune if needed
# This for tfidf or countvectorizer method only
def train_fine_tune(models, df, X, y, params_list, tsize=0.20, rstate=45, vec_type="tfidf",
                    search_type="grid", random_niter=4):
    """
    Trains specific machine learning algorithms, also fine tunes the algorithms, and if the fine tuned version of the algorithms peforms better then it returns those models, otherwise returns the base version of the algorithms.
    Args:
        models: list of machine learning algorithms
        df: pandas dataframe
        X: string of feature column, description or synopsis column here
        y: string of label column
        param_list: a list of parameter grids that we use in search techniques
        tsize: test size for train test split
        rstate: random state for train test split
        vec_type: default is "tfidf", for TfIdfVectorizer, and "count" for CountVectorizer
        search_type: default is "grid", for GridSearchCV, and "random" for "RandomizedSearchCv"
        random_niter=4: n_iter value for RandomizedSearchCV. Set this when search_type is set "random"
    Returns:
        filt_models: best models out of both the base algorithms and fine tuned algorithms
    """

    try:
        if len(models) != len(params_list):
            raise FineTuneError("The models and their corresponding params should be of the same number, and should be in proper order!")
    except Exception as e:
        print(f"Exception: {e}")
        return # Terminate

    syn = df[X]
    lab = df[y]
    X_train, X_test, y_train, y_test = train_test_split(
        syn, lab, test_size=tsize, random_state=rstate
    )
    try:
        if vec_type == "tfidf":
            vec = TfidfVectorizer()
        elif vec_type == "count":
            vec = CountVectorizer()
        else:
            raise FineTuneError("Enter a valid string for choosing vectorizer!")
    except Exception as e:
        print(f"Exception occured: {e}")
        return # Terminate
    
    spX_train = vec.fit_transform(X_train)
    spX_test = vec.transform(X_test)

    perf_base = {}
    perf_tune = {}
    base_models = []
    tune_models = []
    filt_models = [] # filtered models

    print("\n\n---> Base models:")
    for model in models:
        model.fit(spX_train, y_train)
        preds = model.predict(spX_test)
        creport = classification_report(y_test, preds)
        acc = accuracy_score(y_test, preds)
        print(f"\nModel: {model}")
        print(f"\nClassification report:\n{creport}")
        print(f"\nOverall accuracy: {acc}")
        
        base_models.append(model)
        perf_base[str(model)] = acc
    
    try:
        if not (search_type == "grid" or search_type == "random"):
            raise FineTuneError("Enter a valid string for chooisng the search technique!")
    except Exception as e:
        print(f"Exception: {e}")
        return # Terminate
    
    print("\nPlease wait for some time, search technique is finding the best hyper parameters...")
    print(f"\n\n---> Tuning models:")
    for ind, model in enumerate(models):
        if search_type == "grid":
            model_tune = GridSearchCV(
                estimator=model,
                param_grid=params_list[ind],
                scoring="accuracy",
                n_jobs=4, cv=5,

            )
        elif search_type == "random":
            model_tune = RandomizedSearchCV(
                estimator=model,
                param_distributions=params_list[ind],
                scoring="accuracy",
                cv=5,
                n_jobs=4,
                n_iter=random_niter
            )
               
        model_tune.fit(spX_train, y_train)
        preds_tune = model_tune.predict(spX_test)
        acc_tune = accuracy_score(y_test, preds_tune)
        creport_tune = classification_report(y_test, preds_tune)
        print(f"\nModel: {model_tune}")
        print(f"\nClassification report:\n{creport_tune}")
        print(f"\nOverall accuracy: {acc_tune}")

        tune_models.append(model_tune)
        perf_tune[str(model)] = acc_tune

    print("---> Comparing and selecting the best algorithms:")
    model_str = list(perf_base.keys())
    i = 0
    while i < len(models):
        if perf_tune[model_str[i]] > perf_base[model_str[i]]:
            print(f"-> Tuned version of {model_str[i]} performs better.")
            filt_models.append(tune_models[i])
        else:
            print(f"-> Base version of {model_str[i]} performs better.")
            filt_models.append(base_models[i])
        i += 1

    return filt_models

if __name__ == "__main__":
    # Testing the function
    df = pd.read_excel("D:/projects-aiml/Binge-Trend/ml_backend/data_sets/rotten_tomatoes_100.xlsx")
    df = df[(df["label"]=="romantic") | (df["label"]=="horror")]
    filt_models = train_fine_tune(
        models=[
            AdaBoostClassifier(algorithm="SAMME", random_state=42),
            LogisticRegression(max_iter=10000, random_state=42)
        ],
        df=df,
        X="synopsis", y="label",
        params_list=[
            {
                "n_estimators":[50,100,150,200],
            },
            {
                "C":np.logspace(1,10,20), # Intensity of l1,l2
                "class_weight":["balanced", None],
                "solver":["saga"], # For performing multiclass classification
                "penalty":["l1","l2"]
            }
        ],
        tsize=0.20,
        rstate=45,
        vec_type="tfidf",
        search_type="random",
        random_niter=4
    )

    print("\nThese are the filtered models:")
    for ind, mod in enumerate(filt_models):
        print(f"Here's model number - {ind + 1}")
        print(mod)
    

    










    




    








