import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Datasets for anime
adf1 = pd.read_excel("datasets/anime_imbd.xlsx")
adf2 = pd.read_excel("datasets/anime_crunchyroll.xlsx")

def lower_words_synopsis(data_sets):
    new_datasets = []
    for data in data_sets:
        data["synopsis"] = data["synopsis"].str.lower()
        new_datasets.append(data)

    return new_datasets

# Function for testing multiple datasets
def data_tester(models, data_sets, report_path, 
                acc=True, creport=True, cfmat=True, tsize=0.20, rstate=45, 
                dataset_names=None, vec_type="count"):
    """
    Generates reports on multiple datasets to find the best quality dataset.

    Args:
        models: list of scikit-learn models
        data_sets: list of datasets
        report_path: path where the txt file of report will be generated
        acc: If true then accuracy score will be calculated
        creport: If true then classification report is generated
        cfmat: If true then classification report will be generated
        tsize: (int) Test size for train test split
        rstate: (int) Random integer value for reproducibility
        dataset_names: List of the names of the datasets
        vec_type: Type of vectorizer, either CountVectorizer, or TfidfVectorizer
    
    Returns:
        None: Returns nothing, but generates text files of report at the provided path
    """

    if vec_type == "count":
        vec = CountVectorizer(stop_words="english")
    elif vec_type == "tfidf":
        vec = TfidfVectorizer(stop_words="english")
    else:
        print("Enter a valid string for choosing the vectorizer!")
        return
        
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
                ("vec", vec), 
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
    df1, df2 = lower_words_synopsis([df1, df2])
    data_tester(
        report_path="./reports/final_report_movies.txt",
        acc=True,
        creport=True,
        cfmat=True,
        rstate=45,
        tsize=0.20,
        data_sets=[df1, df2], 
        models=[
            GradientBoostingClassifier(),
            AdaBoostClassifier(algorithm="SAMME"),
            LogisticRegression(),
            DecisionTreeClassifier(),
            MultinomialNB(),
            RandomForestClassifier()],                                          
        dataset_names=["IMBD", "ROTTEN TOMATOES"],
        vec_type="tfidf" # use TFIDF because it performed better!
    )

    # Testing only two datasets
    # Imbd & Crunchyroll
    adf1, adf2 = lower_words_synopsis([adf1, adf2])
    data_tester(
        report_path="./reports/final_anime_report.txt",
        acc=True,
        creport=True,
        cfmat=True,
        rstate=45,
        tsize=0.20,
        data_sets=[adf1, adf2], 
        models=[
            GradientBoostingClassifier(),
            AdaBoostClassifier(algorithm="SAMME"),
            LogisticRegression(),
            DecisionTreeClassifier(),
            MultinomialNB(),
            RandomForestClassifier()],                                          
        dataset_names=["IMBD", "CRUNCHYROLL"],
        vec_type="tfidf"
    )

# Note:-
# Make sure you delete the text files at the folder './reports/' before running this script! (Because the file writing is being set in append mode)
# Important Note:-
# Proper text preprocessing is not done here. Because we are only finding the best quality data here.