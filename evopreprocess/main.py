# import warnings filter
from multiprocessing.spawn import freeze_support
from warnings import simplefilter
# ignore all future warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

simplefilter(action='ignore', category=FutureWarning)
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from EvoPreprocess.feature_selection import EvoFeatureSelection
import NiaPy.algorithms.basic as nia
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
import pandas as pd

# Set the random seed for the reproducibility
rs = 1000


def ex1():
    # Load regression data
    dataset = load_boston()

    # Split the dataset to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.33, random_state=rs)

    # Train the decision tree model
    model = DecisionTreeRegressor(random_state=rs)
    model.fit(X_train, y_train)

    # Print the results: shape of the original dataset and the accuracy of decision tree regressor on original data
    print(X_train.shape, mean_squared_error(y_test, model.predict(X_test)), sep=': ')

    # Sample the data with random_seed set
    evo = EvoFeatureSelection(n_jobs=1, n_runs=1, evaluator=model, optimizer=nia.GreyWolfOptimizer, random_seed=rs)
    X_train_new = evo.fit_transform(X_train, y_train)

    # Fit the decision tree model
    model.fit(X_train_new, y_train)

    # Keep only selected feature on test set
    X_test_new = evo.transform(X_test)

    # Print the results: shape of the original dataset and the MSE of decision tree regressor on original data
    print(X_train_new.shape, mean_squared_error(y_test, model.predict(X_test_new)), sep=': ')


def ex2():
    path = "../data/cyberbully.xlsx"
    data = pd.read_excel(path)
    texts = data["text"]
    labels = data["cyberbully"]
    v = CountVectorizer(min_df=120, ngram_range=(3, 3), analyzer='char')
    term_doc_matrix = v.fit_transform(texts)
    term_doc_matrix = term_doc_matrix.toarray()
    print("# of samples: " + str(term_doc_matrix.shape[0]))
    print("# of features: " + str(term_doc_matrix.shape[1]))

    feature_names = v.get_feature_names()
    selector = SelectKBest(chi2, k=10)
    X_new = selector.fit_transform(term_doc_matrix, labels)
    mask = selector.get_support(indices=True)
    scores = selector.scores_
    for idx, feature in zip(mask, feature_names):
        if bool:
            print("score: {} score :{}".format(scores[idx], feature))

    print("*" * 100)
    evo = EvoFeatureSelection(n_runs=4, n_folds=2, evaluator=LogisticRegression(), optimizer=nia.ParticleSwarmOptimization, random_seed=rs)
    x_train_new = evo.fit_transform(term_doc_matrix, labels)
    mask = evo.get_support(indices=True)
    scores = evo.scores_
    for idx, feature in zip(mask, feature_names):
        if bool:
            print("score: {} score :{}".format(scores[idx], feature))


if __name__ == '__main__':
    freeze_support()
    ex2()

