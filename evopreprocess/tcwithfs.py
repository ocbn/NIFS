from multiprocessing.spawn import freeze_support
from warnings import simplefilter

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

simplefilter(action='ignore', category=FutureWarning)  # ignore all future warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import NiaPy.algorithms.basic as nia
from sklearn import metrics
from statistics import mean
from evopreprocess.EvoFeatureSelection import EvoFeatureSelection
import time
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


rs = 123  # random seed
n_folds = 5

# min_df = 2
# path = "../data/cyberbully.xlsx"
# data = pd.read_excel(path)
# texts = data["text"]
# labels = data["label"]

min_df = 3
path = "../data/ttc_digits_removed.xlsx"
data = pd.read_excel(path)
texts = data["text"]
labels = data["label"]

classifiers = [
            ('LR', LogisticRegression(solver='lbfgs', max_iter=300, random_state=rs)),
            ('SVM', SVC(kernel='linear', random_state=rs)),
            ('NB', GaussianNB()),
            ('MNB', MultinomialNB()),
            ('DT', DecisionTreeClassifier(random_state=rs)),
            ('RF', RandomForestClassifier(n_estimators=100, random_state=rs))
        ]

vectorizers = [
    # ('2tf-idf', TfidfVectorizer(min_df=min_df, ngram_range=(2, 2), analyzer='char')),
    # ('2tf', CountVectorizer(min_df=min_df, ngram_range=(2, 2), analyzer='char')),
    # ('2binary', CountVectorizer(min_df=min_df, ngram_range=(2, 2), analyzer='char', binary=True)),
    # ('bow_tf-idf', TfidfVectorizer(min_df=min_df)),
    # ('bow_tf', CountVectorizer(min_df=min_df)),
    # ('bow_binary', CountVectorizer(min_df=min_df, binary=True)),
    ('3tf-idf', TfidfVectorizer(min_df=min_df, ngram_range=(3, 3), analyzer='char')),
    # ('3tf', CountVectorizer(min_df=min_df, ngram_range=(3, 3), analyzer='char')),
    # ('3binary', CountVectorizer(min_df=min_df, ngram_range=(3, 3), analyzer='char', binary=True))
]


def model1():
    optimizers = [
        ('GA', nia.GeneticAlgorithm),
        ('GWO', nia.GreyWolfOptimizer),
        ('SCA', nia.SineCosineAlgorithm),
        ('ABC', nia.ArtificialBeeColonyAlgorithm),
        ('BAT', nia.BatAlgorithm),
        ('CKO', nia.CuckooSearch),
        ('DE', nia.DifferentialEvolution),
        ('HS', nia.HarmonySearch),
        ('PSO', nia.ParticleSwarmOptimization),
        ('FA', nia.FireflyAlgorithm)
    ]
    # for c_name, classifier in classifiers:
    info = []
    for v_name, v in vectorizers:
        for o_name, o in optimizers:
            file = open("r" + v_name + "_" + o_name + ".txt", "w+")
            # pipeline = Pipeline(steps=[
            #     ('vectorizer', CountVectorizer(min_df=2, ngram_range=(3, 3), analyzer='char')),
            #     ('to_dense', DenseTransformer()),
            #     ('feature_selection', EvoFeatureSelection(n_runs=4, n_folds=2, evaluator=DecisionTreeClassifier(), optimizer=nia.GeneticAlgorithm, random_seed=rs)),
            #     ('classifier', classifier)])
            #
            # # Fit the pipeline
            # # pipeline.fit(X_train, y_train)
            # cv = StratifiedKFold(n_folds).split(texts, labels)
            # # Print the results: the accuracy of the pipeline
            # accuracy = cross_val_score(pipeline, texts, labels, scoring='accuracy', cv=cv).mean()
            # print("Average acc of " + name + " is: ", accuracy * 100)

            # print("Classifier:{} | Vectorizer: {} | Optimizer: {}".format(c_name, v_name, o_name), end=" | ")
            i1 = "Classifier:{} | Vectorizer: {} | Optimizer: {}".format("LR", v_name, o_name)
            print(i1, end=" | ")
            acc_scores, num_selected = [], []
            term_doc_matrix = v.fit_transform(texts)
            term_doc_matrix = term_doc_matrix.toarray()
            print("# of samples: " + str(term_doc_matrix.shape[0]))
            print("# of features: " + str(term_doc_matrix.shape[1]))
            for train_index, test_index in StratifiedKFold(n_folds).split(term_doc_matrix, labels):
                print("RUNNING...!")
                x_train, x_test = term_doc_matrix[train_index], term_doc_matrix[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                evo = EvoFeatureSelection(n_runs=4, n_folds=2,
                                          evaluator=LogisticRegression(solver='lbfgs', max_iter=300, random_state=rs),
                                          optimizer=o,
                                          random_seed=rs)
                x_train_new = evo.fit_transform(x_train, y_train)
                num_selected.append(x_train_new.shape[1])
                x_test_new = evo.transform(x_test)
                classifier = LogisticRegression(solver='lbfgs', max_iter=300, random_state=rs)
                classifier.fit(x_train_new, y_train)
                predictions = classifier.predict(x_test_new)
                acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
                acc_scores.append(acc)
            i2 = "Average acc:{}".format(mean(acc_scores) * 100)
            print(i2, end=" | ")
            i3 = "# of selected features: {}".format(num_selected)
            print(i3, end="")
            info.append(i1 + " " + i2 + " " + i3)
            print(i1 + " " + i2 + " " + i3 + "\n")
            file.write(i1 + " " + i2 + " " + i3 + "\n")
            print('-' * 100)
            file.close()

    print("*" * 50)
    for inf in info:
        print(inf)


def mi(X, y):
    return mutual_info_classif(X=X, y=y, random_state=rs)


def model2():
    for v_name, v in vectorizers:
        term_doc_matrix = v.fit_transform(texts)
        term_doc_matrix = term_doc_matrix.toarray()
        df = pd.DataFrame(columns=["k", "LR", "SVM", "NB", "MNB", "DT", "RF"])
        # df = pd.DataFrame(columns=["k", "LR"])
        idx = 0
        print("Vectorizer: {}".format(v_name))
        print('-' * 100)
        # for k in range(10, term_doc_matrix.shape[1], 10):
        for k in [2750]:
            acc_scores = {"LR": [], "SVM": [], "NB": [], "MNB": [], "DT": [], "RF": []}
            # acc_scores = {"LR": []}
            for train_index, test_index in StratifiedKFold(n_folds).split(term_doc_matrix, labels):
                x_train, x_test = term_doc_matrix[train_index], term_doc_matrix[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                # selector = SelectKBest(chi2, k=k)  # random state algoritma dogası geregi yok
                # selector = SelectKBest(mi, k=k) # random state var. (Biz asagıdakini kullandık)
                selector = SelectKBest(mutual_info_classif, k=k)  # random state yok, her run icin farklı sonuc gelir
                x_train_new = selector.fit_transform(x_train, y_train)
                x_test_new = selector.transform(x_test)
                for c_name, c in classifiers:
                    c.fit(x_train_new, y_train)
                    predictions = c.predict(x_test_new)
                    acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
                    acc_scores[c_name].append(acc)
            for c_key, res in acc_scores.items():
                avg_acc = mean(res) * 100
                print("\t[{} | k: {} | avg_acc: {}]".format(c_key, k, avg_acc))
                df.at[idx, "k"] = k
                df.at[idx, c_key] = avg_acc
            print('-' * 100)
            idx += 1
        print('-' * 100)
        df.to_excel("ttc_" + c_name + "_" + v_name + "_mutual.xlsx")


def model3():
    for v_name, v in vectorizers:
        tdm = v.fit_transform(texts)
        tdm = tdm.toarray()
        print('-' * 100)
        print("Vectorizer: {} | # of samples: {} | # of features: {}".
              format(v_name, str(tdm.shape[0]), str(tdm.shape[1])))
        print('-' * 100)
        # cross_val_score ile aynı sonucu veriyor n_fold sonucunda!
        # f_score = cross_val_score(classifier, term_doc_matrix, labels, scoring='accuracy', cv=n_folds).mean() * 100
        # print("Average acc of " + name + " is: ", f_score)
        for name, classifier in classifiers:
            acc_scores = []
            for train_index, test_index in StratifiedKFold(n_folds).split(tdm, labels):
                x_train, x_test = tdm[train_index], tdm[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                classifier.fit(x_train, y_train)
                predictions = classifier.predict(x_test)
                acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
                acc_scores.append(acc)
            print("Classifier: {}, average acc: {}".format(name, mean(acc_scores) * 100))
            print('-' * 100)


def filter_based_execution_time():
    v = TfidfVectorizer(min_df=min_df, ngram_range=(3, 3), analyzer='char')
    term_doc_matrix = v.fit_transform(texts)
    term_doc_matrix = term_doc_matrix.toarray()
    acc_scores, f_time, c_time, f_and_c_time = [], [], [], []
    idx = 0
    for train_index, test_index in StratifiedKFold(n_folds).split(term_doc_matrix, labels):
        print("Fold: {}".format(idx + 1))
        x_train, x_test = term_doc_matrix[train_index], term_doc_matrix[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        fs_time = time.time()
        # selector = SelectKBest(mi, k=5810)
        selector = SelectKBest(chi2, k=4530)
        x_train_new = selector.fit_transform(x_train, y_train)
        x_test_new = selector.transform(x_test)
        fe_time = time.time()
        f_time.insert(idx, fe_time - fs_time)
        cs_time = time.time()
        classifier = LogisticRegression(solver='lbfgs', max_iter=300, random_state=rs)
        classifier.fit(x_train_new, y_train)
        predictions = classifier.predict(x_test_new)
        ce_time = time.time()
        c_time.insert(idx, ce_time - cs_time)
        f_and_c_time.insert(idx, f_time[idx] + c_time[idx])
        acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
        acc_scores.insert(idx, acc)
        idx += 1
    for k in range(n_folds):
        print("Fold {}: Acc: {} FS Time: {} C Time: {} FS + C Time: {}".format(k, acc_scores[k], f_time[k], c_time[k], f_and_c_time[k]))
    print("Average FS Time: {}".format(mean(f_time)))
    print("Average C Time: {}".format(mean(c_time)))
    print("Average FS + C Time: {}".format(mean(f_and_c_time)))
    print("Average Acc: {}".format(mean(acc_scores)))
    print("# of selected features in each fold: {}".format(selector.k))


def nature_inspired_execution_time():
    optimizers = [
        # ('GA', nia.GeneticAlgorithm),
        # ('GWO', nia.GreyWolfOptimizer),
        # ('SCA', nia.SineCosineAlgorithm),
        ('ABC', nia.ArtificialBeeColonyAlgorithm),
        # ('BAT', nia.BatAlgorithm),
        # ('CKO', nia.CuckooSearch),
        # ('DE', nia.DifferentialEvolution),
        # ('HS', nia.HarmonySearch),
        # ('PSO', nia.ParticleSwarmOptimization),
        # ('FA', nia.FireflyAlgorithm)
    ]
    for v_name, v in vectorizers:
        tdm = v.fit_transform(texts)
        tdm = tdm.toarray()
        print("# of samples: {} # of features: {}".format(str(tdm.shape[0]), str(tdm.shape[1])))
        for o_name, o in optimizers:
            print("Classifier:{} | Vectorizer: {} | Optimizer: {}".format("LR", v_name, o_name))
            acc_scores, f_time, c_time, f_and_c_time, num_selected = [], [], [], [], []
            f_idx = 0
            file = open("cyber_" + v_name + "_lr_inner_2_fold" + o_name + ".txt", "w+")
            for train_index, test_index in StratifiedKFold(n_folds).split(tdm, labels):
                print("Fold: {}".format(f_idx + 1))
                x_train, x_test = tdm[train_index], tdm[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                evo_start = time.time()
                evo = EvoFeatureSelection(n_runs=4, n_folds=2, evaluator=LogisticRegression(solver='lbfgs', max_iter=300, random_state=rs), optimizer=o, random_seed=rs)
                                          # optimizer_settings={'NP': 100})
                # nGEN (int): Maximum number of algorithm iterations/generations.
                # nFES (int): Maximum number of function evaluations. Default 1000
                # NP: Population size.
                x_train_new = evo.fit_transform(x_train, y_train)
                evo_end = time.time()
                evo_time = evo_end - evo_start
                f_time.insert(f_idx, evo_time)
                num_selected.insert(f_idx, x_train_new.shape[1])
                x_test_new = evo.transform(x_test)
                cl_start = time.time()
                classifier = LogisticRegression(solver='lbfgs', max_iter=300, random_state=rs)
                classifier.fit(x_train_new, y_train)
                predictions = classifier.predict(x_test_new)
                cl_end = time.time()
                cl_time = cl_end - cl_start
                c_time.insert(f_idx, cl_time)
                acc_scores.insert(f_idx, metrics.accuracy_score(y_true=y_test, y_pred=predictions))
                f_and_c_time.insert(f_idx, evo_time + cl_time)
                file.write("Fold {}: Acc: {} FS Time: {} C Time: {} FS + C Time: {} FC:{}\n".
                           format(f_idx + 1, acc_scores[f_idx], f_time[f_idx], c_time[f_idx], f_and_c_time[f_idx], num_selected[f_idx]))
                f_idx += 1
            file.write("Average FS Time: {}\n".format(mean(f_time)))
            file.write("Average C Time: {}\n".format(mean(c_time)))
            file.write("Average FS + C Time: {}\n".format(mean(f_and_c_time)))
            file.write("Average Acc: {}\n".format(mean(acc_scores)))
            file.write("Average FC: {}\n".format(mean(num_selected)))
            file.close()
            for k in range(n_folds):
                print("Fold {}: Acc: {} FS Time: {} C Time: {} FS + C Time: {} FC:{}".
                      format(k, acc_scores[k], f_time[k], c_time[k], f_and_c_time[k], num_selected[k]))
            print("Average FS Time: {}".format(mean(f_time)))
            print("Average C Time: {}".format(mean(c_time)))
            print("Average FS + C Time: {}".format(mean(f_and_c_time)))
            print("Average Acc: {}".format(mean(acc_scores)))
            print("Average FC: {}".format(mean(num_selected)))
            print()


if __name__ == '__main__':
    freeze_support()
    # model1()
    # filter_based_execution_time()
    # nature_inspired_execution_time()
    model3()
    # model2()
    # model2_2()