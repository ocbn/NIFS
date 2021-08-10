from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)  # ignore all future warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from statistics import mean
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import NiaPy.algorithms.basic as nia
from evopreprocess.EvoFeatureSelection import EvoFeatureSelection
import time


rs = 123  # random seed
n_folds = 5
classifiers = [
        ('LR', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)),
        ('SVM', SVC(kernel='linear', random_state=rs)),
        ('NB', GaussianNB()),
        ('MNB', MultinomialNB()),
        ('DT', DecisionTreeClassifier(random_state=rs)),
        ('RF', RandomForestClassifier(n_estimators=100, random_state=rs))
    ]


def mi(X, y):
    return mutual_info_classif(X=X, y=y, random_state=rs)


def model2_2():
    paths = {2: "../data/cyberbully.xlsx", 3: "../data/ttc_digits_removed.xlsx"}
    for mdf, p in paths.items():
        data_name = "cyber" if mdf == 2 else "ttc"
        data = pd.read_excel(p)
        texts = data["text"]
        labels = data["label"]
        vector = [
            ('3tf-idf', TfidfVectorizer(min_df=mdf, ngram_range=(3, 3), analyzer='char')),
            ('3tf', CountVectorizer(min_df=mdf, ngram_range=(3, 3), analyzer='char')),
            ('3binary', CountVectorizer(min_df=mdf, ngram_range=(3, 3), analyzer='char', binary=True))
        ]
        for v_name, v in vector:
            term_doc_matrix = v.fit_transform(texts)
            term_doc_matrix = term_doc_matrix.toarray()
            df = pd.DataFrame(columns=["k", "LR", "SVM", "NB", "MNB", "DT", "RF"])
            idx = 0
            print("Weighting: {}, Data: {}, Min. DF: {}".format(v_name, data_name, mdf))
            print('-' * 100)
            for k in range(10, term_doc_matrix.shape[1], 10):
                acc_scores = {"LR": [], "SVM": [], "NB": [], "MNB": [], "DT": [], "RF": []}
                for train_index, test_index in StratifiedKFold(n_folds).split(term_doc_matrix, labels):
                    x_train, x_test = term_doc_matrix[train_index], term_doc_matrix[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    selector = SelectKBest(mi, k=k)  # random state var.
                    x_train_new = selector.fit_transform(x_train, y_train)
                    x_test_new = selector.transform(x_test)
                    for c_name, c in classifiers:
                        c.fit(x_train_new, y_train)
                        predictions = c.predict(x_test_new)
                        acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
                        acc_scores[c_name].append(acc)
                for c_key, res in acc_scores.items():
                    avg_acc = mean(res)
                    print("\t[{} | k: {} | avg_acc: {}]".format(c_key, k, avg_acc))
                    df.at[idx, "k"] = k
                    df.at[idx, c_key] = avg_acc
                print('-' * 100)
                idx += 1
            print('-' * 100)
            df.to_excel(data_name + "_" + v_name + "_mi.xlsx")


def compare_nifs_wrt_population_size():
    optimizers = [
        # ('ABC', nia.ArtificialBeeColonyAlgorithm),
        ('CS', nia.CuckooSearch),
        ('FA', nia.FireflyAlgorithm)
    ]
    # paths = {2: "../data/cyberbully.xlsx", 3: "../data/ttc_digits_removed.xlsx"}
    paths = {3: "../data/ttc_digits_removed.xlsx"}
    for mdf, p in paths.items():
        data_name = "cyber" if mdf == 2 else "ttc"
        data = pd.read_excel(p)
        texts = data["text"]
        labels = data["label"]
        vector = TfidfVectorizer(min_df=mdf, ngram_range=(3, 3), analyzer='char')
        tdm = vector.fit_transform(texts)
        tdm = tdm.toarray()
        print("# of samples: {} # of features: {}".format(str(tdm.shape[0]), str(tdm.shape[1])))
        # NP = [10, 30, 50, 100, 150, 200, 300]
        NP = [300]
        # NP = [500, 750, 1000]
        for np in NP:
            for o_name, o in optimizers:
                print("Classifier:{} | Vectorizer: {} | Optimizer: {}, NP: {}".format("LR", "tf-idf", o_name, np))
                acc_scores, f_time, c_time, f_and_c_time, num_selected = [], [], [], [], []
                f_idx = 0
                file = open(data_name + "_tf_idf_lr_inner_2_fold" + o_name + "_" + str(np) + ".txt", "w+")
                for train_index, test_index in StratifiedKFold(n_folds).split(tdm, labels):
                    print("Fold: {}".format(f_idx + 1))
                    x_train, x_test = tdm[train_index], tdm[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    evo_start = time.time()
                    evo = EvoFeatureSelection(n_runs=4, n_folds=2, evaluator=LogisticRegression(solver='lbfgs', max_iter=300, random_state=rs), optimizer=o, random_seed=rs,
                                              optimizer_settings={'NP': np})
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


def model3_with_computation_time():
    paths = {2: "../data/cyberbully.xlsx", 3: "../data/ttc_digits_removed.xlsx"}
    for mdf, p in paths.items():
        data_name = "cyber" if mdf == 2 else "ttc"
        data = pd.read_excel(p)
        texts = data["text"]
        labels = data["label"]
        vector = TfidfVectorizer(min_df=mdf, ngram_range=(3, 3), analyzer='char')
        tdm = vector.fit_transform(texts)
        tdm = tdm.toarray()
        print("# of samples: {} # of features: {}".format(str(tdm.shape[0]), str(tdm.shape[1])))
        acc_scores, c_time = [], []
        for train_index, test_index in StratifiedKFold(n_folds).split(tdm, labels):
            x_train, x_test = tdm[train_index], tdm[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            c_start = time.time()
            classifier = LogisticRegression(solver='lbfgs', max_iter=300, random_state=rs)
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)
            c_end = time.time()
            c_time.append(c_end - c_start)
            acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
            acc_scores.append(acc)
        print("Data: {}, Classifier: {}, average acc: {}".format(data_name, "LR", mean(acc_scores) * 100))
        print("Average C Time: {}\n".format(mean(c_time)))
        print('-' * 100)


if __name__ == '__main__':
    model2_2()
    # compare_nifs_wrt_population_size()
    # model3_with_computation_time()
