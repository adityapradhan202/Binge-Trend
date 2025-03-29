import pandas as pd
# import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load all the datasets here
df1 = pd.read_excel("datasets/imbd_test_data.xlsx")
df2 = pd.read_excel("datasets/rotten_tomatoes.xlsx")
df3 = pd.read_excel("datasets/synthetic_100.xlsx")

# Renaming some columns - Make sure  the column names are in the right format
# Column names - title, synopsis, label
df1 = df1.rename({"movie_name":"title", "plot":"synopsis"}, axis=1)
df2 = df2.rename({"Title":"title", "Genre":"label"}, axis=1)
df3 = df3.rename({"Title":"title", "Genre":"label", "Synopsis":"synopsis"}, axis=1)

def lower_words_synopsis(data_sets):
    new_datasets = []
    for data in data_sets:
        data["synopsis"] = data["synopsis"].str.lower()
        new_datasets.append(data)

    return new_datasets

# Function for testing multiple datasets
def data_tester(models,
                report_path, 
                acc=True, 
                creport=True, 
                cfmat=True, 
                tsize=0.20, rstate=45, vec=CountVectorizer(stop_words="english"), 
                data_sets=lower_words_synopsis([df1, df2, df3]),
                dataset_names=None):
    """Docstring"""
    
    for ind, data in enumerate(data_sets):
        X, y = data["synopsis"], data["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, 
                                   random_state=rstate)
        
        if dataset_names != None:
            print(f"\n\n------> DATASET NUMBER - {ind + 1}, {dataset_names[ind]}\n\n")
            try:
                with open(report_path, "a") as file:
                    file.write(f"--------> DATASET NUMBER - {ind + 1}, {dataset_names[ind]}\n\n")
            except Exception as e:
                print(f"Some exception occured: {e}")
        else:
            print(f"\n\n------> DATASET NUMBER - {ind + 1}\n\n")
            try:
                with open(report_path, "a") as file:
                    file.write(f"--------> DATASET NUMBER - {ind + 1}\n\n")
            except Exception as e:
                print(f"Some exception occured: {e}")
            
        for model in models:
            pipe = Pipeline([
                ("cvec", vec), 
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            print(f"Model name - {model}")
            if acc == True:
                acc_score = accuracy_score(y_true=y_test, y_pred=preds)
                print(f"Overall Accuracy: {acc_score}")
            if creport == True:
                print("\nClassification report:")
                clf = classification_report(y_true=y_test, y_pred=preds)
                print(clf)
            if cfmat == True:
                conf_mat = confusion_matrix(y_true=y_test, y_pred=preds)
                print("\nConfusion matrix:")
                print(conf_mat)
                
            try:
                with open(report_path, "a") as file:
                    file.write(f"Model name: {model}\n")
                    file.write(clf)
                    file.write("\n")
                    file.write("\nConfusion matrix:\n")
                    file.write(str(conf_mat) + "\n\n")

            except Exception as e:
                print(f"Some exception occured: {e}")
        # Loop ends here
    print(f"\n---> The report has been saved to the path: {report_path}\n")

if __name__ == "__main__":

    # Only imbd and rotten tomatoes
    # Because synthetic data by GPT is overfitting
    df1, df2 = lower_words_synopsis([df1, df2])
    data_tester(
        report_path="./reports/final_report.txt",
        acc=True,
        creport=True,
        cfmat=True,
        rstate=45,
        tsize=0.20,
        vec=CountVectorizer(stop_words="english"),
        data_sets=[df1, df2], 
        models=[
            GradientBoostingClassifier(),
            AdaBoostClassifier(algorithm="SAMME"),
            LogisticRegression(),
            DecisionTreeClassifier(),
            MultinomialNB(),
            RandomForestClassifier()],
        dataset_names=["IMBD", "ROTTEN TOMATOES"]
    )

# Note:-
# Make sure you delete the 'final_report.txt' file at the path './reports/final_report.txt' before running this script! (Because the file writing is being set in append mode)