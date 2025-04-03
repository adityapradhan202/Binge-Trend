# Function for training specific algorithms, also fine tuning them and comparing to the base form of algorithms...
# Work pending here...

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class FineTuneError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def train_fine_tune(models, df, X, y, params_list, tsize=0.20, rstate=45, vec_type="tfidf", search_type="grid"):
    """Trains specific machine learning algorithms, also fine tunes the algorithms, and if the fine tuned version of the algorithms peforms better then it returns those models, otherwise returns the base version of the algorithms."""

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
        if search_type != "grid" or search_type != "random":
            raise FineTuneError("Enter a valid string for chooisng the string technique!")
    except Exception as e:
        print(f"Exception: {e}")
        return # Terminate
    
    print(f"\n\n---> Tune models:")
    for ind, model in enumerate(models):
        if search_type == "grid":
            model_tune = GridSearchCV(
                estimator=model,
                param_grid=params_list[ind],
                scoring="accuracy",
                n_jobs=4, cv=5
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
    pass
    # Test the function here!

    

    










    




    








