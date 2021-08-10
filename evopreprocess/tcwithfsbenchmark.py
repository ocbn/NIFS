from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
from multiprocessing.spawn import freeze_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from statistics import mean
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
import time
from evopreprocess.EvoFeatureSelection import EvoFeatureSelection
from nltk.corpus import reuters
import NiaPy.algorithms.basic as nia
nltk.download('punkt', quiet=True)
nltk.download('reuters', quiet=True)
mdf = 3  # minimum document frequency
rs = 123  # random seed
n_folds = 5
cachedStopWords = stopwords.words("english")
classifiers = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)),
    ('SVM', SVC(kernel='linear', random_state=rs)),
    ('NB', GaussianNB()),
    ('MNB', MultinomialNB()),
    ('DT', DecisionTreeClassifier(random_state=rs)),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=rs))
]
c_names = [item[0] for item in classifiers]


def linguistic_tokenizer(text):
    min_length = 2
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuations
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens


def basic_tokenizer(text):
    min_length = 2
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuations
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    words = map(lambda word: word.lower(), word_tokenize(text))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, words))
    return " ".join(filtered_tokens)


def get_class_distribution(name, target_names, target_list):
    print("Class distribution for: {}".format(name))
    for i in range(len(target_names)):
        print("{} -> {}".format(target_names[i], target_list.count(i)))


def mi(X, y):
    return mutual_info_classif(X=X, y=y, random_state=rs)


def classify_with_traditional_schemes(x_train, x_test, y_train, y_test, targets, d_name):
    vec_helpers = [
        ('bow_tf-idf', TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_tf', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_binary', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer, binary=True)),
        ('3tf-idf', TfidfVectorizer(min_df=mdf, ngram_range=(3, 3), analyzer='char')),
        ('3tf', CountVectorizer(min_df=mdf, ngram_range=(3, 3), analyzer='char')),
        ('3binary', CountVectorizer(min_df=mdf, ngram_range=(3, 3), analyzer='char', binary=True))
    ]
    for v_name, v in vec_helpers:
        train_test_split_classification(x_train, x_test, y_train, y_test, v_helper=v, v_name=v_name, t_name=targets, d_name=d_name)


def train_test_split_classification(x_train, x_test, y_train, y_test, v_helper, v_name, t_name, d_name):
    file = open(d_name + "_" + v_name + ".txt", mode="w")
    file.write("-" * 100 + "\n")
    file.write("# of instances in train: {} and test: {}\n".format(len(x_train), len(x_test)))
    print("# of instances in train: {} and test: {}\n".format(len(x_train), len(x_test)))
    file.write("Feature model-scheme: {}\n".format(v_name))
    print("Feature model-scheme: {}\n".format(v_name))
    file.write("Minimum document frequency: {}\n".format(mdf))
    file.write("Random state: {}\n".format(rs))
    file.write("-" * 100)
    v_s_time = time.time()
    x_train = v_helper.fit_transform(x_train, y_train)
    x_train = x_train.toarray()
    v_e_time = time.time()
    vec_time = v_e_time - v_s_time
    x_test = v_helper.transform(x_test)
    x_test = x_test.toarray()
    file.write("\nVectorizer's computation time: {}\n".format(vec_time))

    print("# of features: {}".format(x_train.shape[1]))
    file.write("# of features: {}\n".format(x_train.shape[1]))
    cv_predicted, cv_cm = classify(x_train, y_train, x_test, y_test)
    file.write("-" * 100)
    for k, v in cv_predicted.items():
        file.write("\nClassification report for {}:\n\t{}\n".format(k, metrics.classification_report(y_true=y_test, y_pred=v, digits=3)))
        file.write("\nConfusion matrix for {}: \n\t{}\n".format(k, cv_cm[k]))
        file.write("-" * 100)
    file.write("\nTarget names:\n")
    for t_idx in range(len(t_name)):
        file.write("\t{} -> {}\n".format(t_idx, t_name[t_idx]))
    file.close()


def classify(x_train, y_train, x_test, y_test):
    y_predictions = dict.fromkeys(c_names, None)
    confusion_matrices = dict.fromkeys(c_names, None)
    for c_name, c in classifiers:
        print("\tClassifying with {}".format(c_name))
        c.fit(x_train, y_train)
        y_predicted = c.predict(x_test)
        y_predictions[c_name] = y_predicted
        confusion_matrices[c_name] = metrics.confusion_matrix(y_test, y_predicted)
    return y_predictions, confusion_matrices


def get_newsgroup_dataset():
    categories = [
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.windows.x',
        'sci.electronics',
        'sci.space',
        'rec.autos',
        'sci.crypt',
        'sci.med'
    ]
    # categories = None
    news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
    # get_class_distribution("news_train", news_train.target_names, list(news_train.target))
    news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
    # get_class_distribution("news_test", news_test.target_names, list(news_test.target))
    targets = news_train.target_names
    x_train = [basic_tokenizer(str(text).strip().rstrip()) for text in list(news_train.data)]
    x_test = [basic_tokenizer(str(text).strip().rstrip()) for text in list(news_test.data)]

    y_train = news_train.target
    y_test = news_test.target

    sx_train, sx_test = {}, {}
    for i in range(len(x_train)):
        text = x_train[i]
        label = y_train[i]
        if label in sx_train:
            sx_train[label].append(text)
        else:
            sx_train[label] = [text]

    for j in range(len(x_test)):
        text = x_test[j]
        label = y_test[j]
        if label in sx_test:
            sx_test[label].append(text)
        else:
            sx_test[label] = [text]
    x_train, x_test, y_train, y_test = [], [], [], []
    for cat, texts in sx_train.items():
        num_instances = 0
        for text in texts:
            if num_instances < 100:
                x_train.append(text)
                y_train.append(cat)
                num_instances += 1
            else:
                break

    for cat, texts in sx_test.items():
        num_instances = 0
        for text in texts:
            if num_instances < 100:
                x_test.append(text)
                y_test.append(cat)
                num_instances += 1
            else:
                break
    get_class_distribution("news_train", targets, y_train)
    get_class_distribution("news_test", targets, y_test)
    return x_train, x_test, y_train, y_test, targets


def get_r8_dataset():
    categories = {
        0: 'acq',
        1: 'crude',
        2: 'earn',
        3: 'grain',
        4: 'interest',
        5: 'money-fx',
        6: 'ship',
        7: 'trade'
    }
    target_names = list(categories.values())
    x_train, x_test, y_train, y_test = [], [], [], []
    train_idx, test_idx = 0, 0
    for cat_id, cat in categories.items():
        category_docs = reuters.fileids(cat)
        n_train, n_test = 0, 0
        for doc_id in category_docs:
            if doc_id.startswith("train") and n_train < 80:
                x_train.insert(train_idx, basic_tokenizer(str(reuters.raw(doc_id, ))))
                y_train.insert(train_idx, cat_id)
                train_idx += 1
                n_train += 1
            elif doc_id.startswith("test") and n_test < 80:
                x_test.insert(test_idx, basic_tokenizer(str(reuters.raw(doc_id))))
                y_test.insert(test_idx, cat_id)
                test_idx += 1
                n_test += 1
    get_class_distribution("r8_train", target_names, y_train)
    get_class_distribution("r8_test", target_names, y_test)
    return x_train, x_test, y_train, y_test, target_names


def get_r8_dataset_2():
    categories = {
        0: 'acq',
        1: 'crude',
        2: 'earn',
        3: 'grain',
        4: 'interest',
        5: 'money-fx',
        6: 'ship',
        7: 'trade'
    }
    target_names = list(categories.values())
    x_train, x_test, y_train, y_test = [], [], [], []
    train_idx, test_idx = 0, 0
    for cat_id, cat in categories.items():
        category_docs = reuters.fileids(cat)
        train_docs = [doc for doc in category_docs if doc.startswith("train")]
        test_docs = [doc for doc in category_docs if doc.startswith("test")]
        n_train, n_test = 0, 0
        for doc_id in train_docs:
            if n_train < int(len(train_docs) / 2):
                x_train.insert(train_idx, basic_tokenizer(str(reuters.raw(doc_id, ))))
                y_train.insert(train_idx, cat_id)
                train_idx += 1
                n_train += 1
        for doc_id in test_docs:
            if n_test < int(len(test_docs) / 2):
                x_test.insert(test_idx, basic_tokenizer(str(reuters.raw(doc_id))))
                y_test.insert(test_idx, cat_id)
                test_idx += 1
                n_test += 1
    get_class_distribution("r8_train", target_names, y_train)
    get_class_distribution("r8_test", target_names, y_test)
    return x_train, x_test, y_train, y_test, target_names


def r8_nature_inspired_mh_based_fs(x_tr, x_te, y_train, y_test):
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
    vec_helpers = [
        ('bow_tf-idf', TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_tf', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_binary', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer, binary=True)),
    ]
    for v_name, v in vec_helpers:
        x_train = v.fit_transform(x_tr)
        x_train = x_train.toarray()
        x_test = v.transform(x_te)
        x_test = x_test.toarray()
        print("# of samples: {} # of features: {}".format(str(x_train.shape[0]), str(x_train.shape[1])))
        for o_name, o in optimizers:
            print("Classifier:{} | Vectorizer: {} | Optimizer: {}".format("SVM", v_name, o_name))
            acc_scores, f_time, c_time, f_and_c_time, num_selected = [], [], [], [], []
            file = open("r8_" + v_name + "_svm_inner_2_fold" + o_name + ".txt", "w+")
            evo_start = time.time()
            evo = EvoFeatureSelection(n_runs=1, n_folds=2,
                                      evaluator=SVC(kernel='linear', random_state=rs),
                                      optimizer=o, random_seed=rs)
            print("Fitting evo-feature-selection..")
            x_train_new = evo.fit_transform(x_train, y_train)
            evo_end = time.time()
            evo_time = evo_end - evo_start
            f_time.append(evo_time)
            num_selected.append(x_train_new.shape[1])
            x_test_new = evo.transform(x_test)
            print("Fitting completed..!")
            cl_start = time.time()
            classifier = SVC(kernel='linear', random_state=rs)
            classifier.fit(x_train_new, y_train)
            predictions = classifier.predict(x_test_new)
            cl_end = time.time()
            cl_time = cl_end - cl_start
            c_time.append(cl_time)
            acc_scores.append(metrics.accuracy_score(y_true=y_test, y_pred=predictions))
            f_and_c_time.append(evo_time + cl_time)
            file.write("Acc: {} FS Time: {} C Time: {} FS + C Time: {} FC:{}\n".
                       format(acc_scores[0], f_time[0], c_time[0], f_and_c_time[0], num_selected[0]))
            file.write("Average FS Time: {}\n".format(mean(f_time)))
            file.write("Average C Time: {}\n".format(mean(c_time)))
            file.write("Average FS + C Time: {}\n".format(mean(f_and_c_time)))
            file.write("Average Acc: {}\n".format(mean(acc_scores)))
            file.write("Average FC: {}\n".format(mean(num_selected)))
            file.close()


def n8_nature_inspired_mh_based_fs(x_tr, x_te, y_train, y_test):
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
    vec_helpers = [
        ('bow_tf-idf', TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_tf', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_binary', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer, binary=True)),
    ]
    for v_name, v in vec_helpers:
        x_train = v.fit_transform(x_tr)
        x_train = x_train.toarray()
        x_test = v.transform(x_te)
        x_test = x_test.toarray()
        print("# of samples: {} # of features: {}".format(str(x_train.shape[0]), str(x_train.shape[1])))
        for o_name, o in optimizers:
            print("Classifier:{} | Vectorizer: {} | Optimizer: {}".format("LR", v_name, o_name))
            acc_scores, f_time, c_time, f_and_c_time, num_selected = [], [], [], [], []
            file = open("n8_" + v_name + "_svm_inner_2_fold" + o_name + ".txt", "w+")
            evo_start = time.time()
            evo = EvoFeatureSelection(n_runs=1, n_folds=2,
                                      evaluator=LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs),
                                      optimizer=o, random_seed=rs)
            print("Fitting evo-feature-selection..")
            x_train_new = evo.fit_transform(x_train, y_train)
            evo_end = time.time()
            evo_time = evo_end - evo_start
            f_time.append(evo_time)
            num_selected.append(x_train_new.shape[1])
            x_test_new = evo.transform(x_test)
            print("Fitting completed..!")
            cl_start = time.time()
            classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)
            classifier.fit(x_train_new, y_train)
            predictions = classifier.predict(x_test_new)
            cl_end = time.time()
            cl_time = cl_end - cl_start
            c_time.append(cl_time)
            acc_scores.append(metrics.accuracy_score(y_true=y_test, y_pred=predictions))
            f_and_c_time.append(evo_time + cl_time)
            file.write("Acc: {} FS Time: {} C Time: {} FS + C Time: {} FC:{}\n".
                       format(acc_scores[0], f_time[0], c_time[0], f_and_c_time[0], num_selected[0]))
            file.write("Average FS Time: {}\n".format(mean(f_time)))
            file.write("Average C Time: {}\n".format(mean(c_time)))
            file.write("Average FS + C Time: {}\n".format(mean(f_and_c_time)))
            file.write("Average Acc: {}\n".format(mean(acc_scores)))
            file.write("Average FC: {}\n".format(mean(num_selected)))
            file.close()


def ex_on_r8_dataset_with_fs(x_tr, x_tst, y_train, y_test):
    vec_helpers = [
        # ('bow_tf-idf', TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        # ('bow_tf', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_binary', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer, binary=True)),
    ]
    for v_name, v in vec_helpers:
        x_train = v.fit_transform(x_tr)
        x_train = x_train.toarray()
        x_test = v.transform(x_tst)
        x_test = x_test.toarray()
        df = pd.DataFrame(columns=["k", "SVM"])
        print("Vectorizer: {}".format(v_name))
        print('-' * 100)
        idx = 0
        for k in range(10, x_train.shape[1], 10):
            acc_scores = {"SVM": []}
            # selector = SelectKBest(chi2, k=k)  # random state algoritma dogası geregi yok
            selector = SelectKBest(mi, k=k) # random state var. (Biz asagıdakini kullandık)
            x_train_new = selector.fit_transform(x_train, y_train)
            x_test_new = selector.transform(x_test)
            c = SVC(kernel='linear', random_state=rs)
            c.fit(x_train_new, y_train)
            predictions = c.predict(x_test_new)
            acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
            acc_scores["SVM"].append(acc)
            for c_key, res in acc_scores.items():
                avg_acc = mean(res) * 100
                print("\t[{} | k: {} | avg_acc: {}]".format(c_key, k, avg_acc))
                df.at[idx, "k"] = k
                df.at[idx, c_key] = avg_acc
            idx += 1
            print('-' * 100)
        print('-' * 100)
        df.to_excel("r8_svm_" + v_name + "_mi.xlsx")


def ex_on_n8_dataset_with_fs(x_tr, x_tst, y_train, y_test):
    vec_helpers = [
        ('bow_tf-idf', TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_tf', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)),
        ('bow_binary', CountVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer, binary=True)),
    ]
    for v_name, v in vec_helpers:
        x_train = v.fit_transform(x_tr)
        x_train = x_train.toarray()
        x_test = v.transform(x_tst)
        x_test = x_test.toarray()
        df = pd.DataFrame(columns=["k", "LR"])
        print("Vectorizer: {}".format(v_name))
        print('-' * 100)
        idx = 0
        for k in range(10, x_train.shape[1], 10):
            acc_scores = {"LR": []}
            selector = SelectKBest(chi2, k=k)  # random state algoritma dogası geregi yok
            # selector = SelectKBest(mi, k=k) # random state var. (Biz asagıdakini kullandık)
            x_train_new = selector.fit_transform(x_train, y_train)
            x_test_new = selector.transform(x_test)
            c = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)
            c.fit(x_train_new, y_train)
            predictions = c.predict(x_test_new)
            acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
            acc_scores["LR"].append(acc)
            for c_key, res in acc_scores.items():
                avg_acc = mean(res) * 100
                print("\t[{} | k: {} | avg_acc: {}]".format(c_key, k, avg_acc))
                df.at[idx, "k"] = k
                df.at[idx, c_key] = avg_acc
            idx += 1
            print('-' * 100)
        print('-' * 100)
        df.to_excel("n8_lr_" + v_name + "_chi.xlsx")


def filter_based_execution_time(x_tr, x_tst, y_train, y_test):
    v = TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)
    x_train = v.fit_transform(x_tr)
    x_train = x_train.toarray()
    x_test = v.transform(x_tst)
    x_test = x_test.toarray()
    acc_scores, f_time, c_time, f_and_c_time = [], [], [], []
    fs_time = time.time()
    # selector = SelectKBest(mi, k=3130)
    selector = SelectKBest(chi2, k=1250)
    x_train_new = selector.fit_transform(x_train, y_train)
    x_test_new = selector.transform(x_test)
    fe_time = time.time()
    f_time.insert(0, fe_time - fs_time)
    cs_time = time.time()
    # classifier = SVC(kernel='linear', random_state=rs)
    classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)
    classifier.fit(x_train_new, y_train)
    predictions = classifier.predict(x_test_new)
    ce_time = time.time()
    c_time.insert(0, ce_time - cs_time)
    f_and_c_time.insert(0, f_time[0] + c_time[0])
    acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
    acc_scores.insert(0, acc)
    print("Acc: {} FS Time: {} C Time: {} FS + C Time: {}".format(acc_scores[0], f_time[0], c_time[0], f_and_c_time[0]))
    print("Average FS Time: {}".format(mean(f_time)))
    print("Average C Time: {}".format(mean(c_time)))
    print("Average FS + C Time: {}".format(mean(f_and_c_time)))
    print("Average Acc: {}".format(mean(acc_scores)))
    print("# of selected features in each fold: {}".format(selector.k))


if __name__ == '__main__':
    # x_train, x_test, y_train, y_test, targets = get_r8_dataset()
    # filter_based_execution_time(x_train, x_test, y_train, y_test)
    # classify_with_traditional_schemes(x_train, x_test, y_train, y_test, targets, "r8")
    # x_train, x_test, y_train, y_test, targets = get_newsgroup_dataset()
    # classify_with_traditional_schemes(x_train, x_test, y_train, y_test, targets, "n8")
    # filter_based_execution_time(x_train, x_test, y_train, y_test)

    # freeze_support()
    x_train, x_test, y_train, y_test, _ = get_r8_dataset()
    r8_nature_inspired_mh_based_fs(x_train, x_test, y_train, y_test)
    # x_train, x_test, y_train, y_test, _ = get_newsgroup_dataset()
    # n8_nature_inspired_mh_based_fs(x_train, x_test, y_train, y_test)
    # ex_on_r8_dataset_with_fs(x_train, x_test, y_train, y_test)
    # ex_on_n8_dataset_with_fs(x_train, x_test, y_train, y_test)

    # ex_on_newsgroup_dataset_with_fs()
