import copy
from multiprocessing.spawn import freeze_support
import numpy as np
from numpy import mean
from sklearn import metrics, preprocessing
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
import time
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import  pandas as pd
from evopreprocess.EvoFeatureSelection import EvoFeatureSelection
from nltk.corpus import reuters
import NiaPy.algorithms.basic as nia


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


nltk.download('punkt', quiet=True)
nltk.download('reuters', quiet=True)
mdf = 5  # minimum document frequency
rs = 123  # random seed
np.random.seed(rs)
n_folds = 5
cachedStopWords = stopwords.words("english")

optimizers = [
    ('ABC', nia.ArtificialBeeColonyAlgorithm),
    # ('GA', nia.GeneticAlgorithm),
    # ('GWO', nia.GreyWolfOptimizer),
    # ('SCA', nia.SineCosineAlgorithm),
    # ('BAT', nia.BatAlgorithm),
    # ('CS', nia.CuckooSearch),
    # ('DE', nia.DifferentialEvolution),
    # ('HS', nia.HarmonySearch),
    # ('PSO', nia.ParticleSwarmOptimization),
    # ('FA', nia.FireflyAlgorithm)
]


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
    for t in target_names:
        print("{}: {}".format(t, target_list.count(t)))


def get_reuters_dataset():
    categories = reuters.categories()
    targets = copy.deepcopy(categories)
    x, y = [], []
    instance_idx = 0
    for cat in categories:
        category_docs = reuters.fileids(cat)
        if len(category_docs) >= 5:
            for doc_id in category_docs:
                x.insert(instance_idx, basic_tokenizer(str(reuters.raw(doc_id, ))))
                y.insert(instance_idx, cat)
                instance_idx += 1
        else:
            targets.remove(cat)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(y)
    get_class_distribution("r8", targets, y)
    return x, labels, targets


def get_subset_of_reuters_dataset():
    categories = reuters.categories()
    targets = copy.deepcopy(categories)
    x, y = [], []
    instance_idx = 0
    for cat in categories:
        category_docs = reuters.fileids(cat)
        if 100 >= len(category_docs) >= 20:
            for doc_id in category_docs:
                x.insert(instance_idx, basic_tokenizer(str(reuters.raw(doc_id, ))))
                y.insert(instance_idx, cat)
                instance_idx += 1
        else:
            targets.remove(cat)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(y)
    get_class_distribution("reuters_subset", targets, y)
    return x, labels, targets


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
    x, y = [], []
    idx = 0
    for cat_id, cat in categories.items():
        category_docs = reuters.fileids(cat)
        for doc_id in category_docs:
            x.insert(idx, basic_tokenizer(str(reuters.raw(doc_id, ))))
            y.insert(idx, target_names[cat_id])
            idx += 1
    get_class_distribution("r8", target_names, y)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(y)
    return x, labels, target_names


def get_newsgroup_dataset():
    news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None)
    news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None)

    x_train = [basic_tokenizer(str(text).strip().rstrip()) for text in list(news_train.data)]
    x_test = [basic_tokenizer(str(text).strip().rstrip()) for text in list(news_test.data)]

    y_train = news_train.target
    y_test = news_test.target

    x = {}
    for i in range(len(x_train)):
        text = x_train[i]
        label = y_train[i]
        if label in x:
            x[label].append(text)
        else:
            x[label] = [text]

    for j in range(len(x_test)):
        text = x_test[j]
        label = y_test[j]
        if label in x:
            x[label].append(text)
        else:
            x[label] = [text]

    x_all, y_all = [], []
    for cat, texts in x.items():
        for text in texts:
            x_all.append(text)
            y_all.append(cat)

    targets = list(x.keys())
    # get_class_distribution("newsgroup", targets, y_all)
    return x_all, y_all, targets


def get_n8_dataset():
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
    news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
    news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)

    x_train = [basic_tokenizer(str(text).strip().rstrip()) for text in list(news_train.data)]
    x_test = [basic_tokenizer(str(text).strip().rstrip()) for text in list(news_test.data)]

    y_train = news_train.target
    y_test = news_test.target

    x = {}
    for i in range(len(x_train)):
        text = x_train[i]
        label = y_train[i]
        if label in x:
            x[label].append(text)
        else:
            x[label] = [text]

    for j in range(len(x_test)):
        text = x_test[j]
        label = y_test[j]
        if label in x:
            x[label].append(text)
        else:
            x[label] = [text]

    x_all, y_all = [], []
    for cat, texts in x.items():
        for text in texts:
            x_all.append(text)
            y_all.append(cat)

    targets = list(x.keys())
    # get_class_distribution("newsgroup", targets, y_all)
    return x_all, y_all, targets


def get_n8_default_train_test_split():
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
        for text in texts:
            x_train.append(text)
            y_train.append(targets[cat])

    for cat, texts in sx_test.items():
        for text in texts:
            x_test.append(text)
            y_test.append(targets[cat])
    get_class_distribution("news_train", targets, y_train)
    get_class_distribution("news_test", targets, y_test)
    return x_train, x_test, y_train, y_test


def get_subset_of_r8_dataset():
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
            if doc_id.startswith("train") and n_train < 20:
                x_train.insert(train_idx, basic_tokenizer(str(reuters.raw(doc_id, ))))
                y_train.insert(train_idx, target_names[cat_id])
                train_idx += 1
                n_train += 1
            elif doc_id.startswith("test") and n_test < 20:
                x_test.insert(test_idx, basic_tokenizer(str(reuters.raw(doc_id))))
                y_test.insert(test_idx, target_names[cat_id])
                test_idx += 1
                n_test += 1
    get_class_distribution("r8_train", target_names, y_train)
    get_class_distribution("r8_test", target_names, y_test)
    return x_train, x_test, y_train, y_test


def get_r8_default_train_test_split():
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
            if doc_id.startswith("train"):
                x_train.insert(train_idx, basic_tokenizer(str(reuters.raw(doc_id, ))))
                y_train.insert(train_idx, target_names[cat_id])
                train_idx += 1
                n_train += 1
            elif doc_id.startswith("test"):
                x_test.insert(test_idx, basic_tokenizer(str(reuters.raw(doc_id))))
                y_test.insert(test_idx, target_names[cat_id])
                test_idx += 1
                n_test += 1
    get_class_distribution("r8_train", target_names, y_train)
    get_class_distribution("r8_test", target_names, y_test)
    le = preprocessing.LabelEncoder()
    y_trn = le.fit_transform(y_train)
    y_tst = le.transform(y_test)
    return x_train, x_test, y_trn, y_tst


def stratified_cv_on_benchmark_data(r8=True, texts=None, labels=None):
    v_name = "tf-idf"
    vec_helper = TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)
    for o_name, o in optimizers:
        if r8:
            file = open("r8_bow_" + v_name + "_svm_inner_2_fold_" + o_name + ".txt", "w+")
        else:
            file = open("n8_bow_" + v_name + "_lr_inner_2_fold" + o_name + ".txt", "w+")
        acc_scores, f_time, c_time, f_and_c_time, num_selected = [], [], [], [], []
        f_macros, f_micros, f_averages = [], [], []
        tdm = vec_helper.fit_transform(texts)
        tdm = tdm.toarray()
        labels = np.array(labels)
        print("# of samples: " + str(tdm.shape[0]))
        print("# of features: " + str(tdm.shape[1]))
        f_idx = 0
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=rs)
        for train_index, test_index in skf.split(tdm, labels):
            print("Fold: {}".format(f_idx + 1))
            x_train, x_test = tdm[train_index], tdm[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            evo_start = time.time()
            if r8:
                evo = EvoFeatureSelection(n_runs=1, n_folds=2,
                                          evaluator=SVC(kernel='linear', random_state=rs),
                                          optimizer=o,
                                          random_seed=rs)
            else:
                evo = EvoFeatureSelection(n_runs=1, n_folds=2,
                                          evaluator=LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs),
                                          optimizer=o,
                                          random_seed=rs)
            x_train_new = evo.fit_transform(x_train, y_train)
            evo_end = time.time()
            evo_time = evo_end - evo_start
            f_time.insert(f_idx, evo_time)
            num_selected.append(x_train_new.shape[1])
            x_test_new = evo.transform(x_test)
            cl_start = time.time()
            if r8:
                classifier = SVC(kernel='linear', random_state=rs)
            else:
                classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)
            classifier.fit(x_train_new, y_train)
            predictions = classifier.predict(x_test_new)
            cl_end = time.time()
            cl_time = cl_end - cl_start
            c_time.insert(f_idx, cl_time)
            acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
            acc_scores.append(acc)
            f_macro = f1_score(y_true=y_test, y_pred=predictions, average='macro')
            f_macros.append(f_macro)
            f_micro = f1_score(y_true=y_test, y_pred=predictions, average='micro')
            f_micros.append(f_micro)
            f_weighted = f1_score(y_true=y_test, y_pred=predictions, average='weighted')
            f_averages.append(f_weighted)
            f_and_c_time.insert(f_idx, evo_time + cl_time)
            file.write("Fold {}: Acc: {} FS Time: {} C Time: {} FS + C Time: {} FC:{}\n".
                       format(f_idx + 1, acc_scores[f_idx], f_time[f_idx], c_time[f_idx], f_and_c_time[f_idx],
                              num_selected[f_idx]))
            print("Fold {}: Acc: {} FS Time: {} C Time: {} FS + C Time: {} FC:{}\n".
                       format(f_idx + 1, acc_scores[f_idx], f_time[f_idx], c_time[f_idx], f_and_c_time[f_idx],
                              num_selected[f_idx]))
            f_idx += 1
        file.write("Average FS Time: {}\n".format(mean(f_time)))
        file.write("Average C Time: {}\n".format(mean(c_time)))
        file.write("Average FS + C Time: {}\n".format(mean(f_and_c_time)))
        file.write("Average Acc: {}\n".format(mean(acc_scores)))
        file.write("Average FC: {}\n".format(mean(num_selected)))
        file.flush()
        file.close()

        for k in range(n_folds):
            print(
                "Fold {}: Acc: {} F_macro: {} F_micro: {} F_weighted: {} FS Time: {} C Time: {} FS + C Time: {} FC:{}".
                format(k, acc_scores[k], f_micros[k], f_micros[k], f_averages[k], f_time[k], c_time[k], f_and_c_time[k],
                       num_selected[k]))
        print("Optimzier: {}".format(o_name))
        print("Average FS Time: {}".format(mean(f_time)))
        print("Average C Time: {}".format(mean(c_time)))
        print("Average FS + C Time: {}".format(mean(f_and_c_time)))
        print("Average Acc: {}".format(mean(acc_scores)))
        print("Average F_macro: {}".format(mean(f_macros)))
        print("Average F_micro: {}".format(mean(f_micros)))
        print("Average F_weighted: {}".format(mean(f_averages)))
        print("Average FC: {}".format(mean(num_selected)))
        print()


def holdout_on_benchmark_data(r8, x_tr, x_te, y_train, y_test):
    v_name = "tf-idf"
    vec_helper = TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)
    for o_name, o in optimizers:
        if r8:
            file = open("r8_bow_" + v_name + "_svm_inner_2_fold_" + o_name + ".txt", "w+")
        else:
            file = open("n8_bow_" + v_name + "_lr_inner_2_fold_" + o_name + ".txt", "w+")
        x_train = vec_helper.fit_transform(x_tr)
        x_train = x_train.toarray()
        x_test = vec_helper.transform(x_te)
        x_test = x_test.toarray()
        print("# of samples: {} # of features: {}".format(str(x_train.shape[0]), str(x_train.shape[1])))
        evo_start = time.time()
        if r8:
            evo = EvoFeatureSelection(
                n_runs=4, n_folds=2,
                evaluator=SVC(kernel='linear', random_state=rs),
                optimizer=o,
                random_seed=rs)
        else:
            evo = EvoFeatureSelection(
                n_runs=4, n_folds=2,
                evaluator=LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs),
                optimizer=o,
                random_seed=rs)
        x_train_new = evo.fit_transform(x_train, y_train)
        evo_end = time.time()
        evo_time = evo_end - evo_start
        f_time = evo_time
        num_selected = x_train_new.shape[1]
        x_test_new = evo.transform(x_test)
        cl_start = time.time()
        if r8:
            classifier = SVC(kernel='linear', random_state=rs)
        else:
            classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)
        classifier.fit(x_train_new, y_train)
        predictions = classifier.predict(x_test_new)
        cl_end = time.time()
        cl_time = cl_end - cl_start
        c_time = cl_time
        acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
        f_macro = f1_score(y_true=y_test, y_pred=predictions, average='macro')
        f_micro = f1_score(y_true=y_test, y_pred=predictions, average='micro')
        f_weighted = f1_score(y_true=y_test, y_pred=predictions, average='weighted')
        f_and_c_time = evo_time + cl_time

        file.write("FS Time: {}\n".format(f_time))
        file.write("C Time: {}\n".format(c_time))
        file.write("FS + C Time: {}\n".format(f_and_c_time))
        file.write("Acc: {}\n".format(acc))
        file.write("Acc: {}\n".format(acc))
        file.write("F_macro: {}\n".format(f_macro))
        file.write("F_micro: {}\n".format(f_micro))
        file.write("F_weighted: {}\n".format(f_weighted))
        file.write("FC (# of selected features): {}\n".format(num_selected))
        file.flush()
        file.close()

        print("FS Time: {}".format(f_time))
        print("C Time: {}".format(c_time))
        print("FS + C Time: {}".format(f_and_c_time))
        print("Acc: {}".format(acc))
        print("F_macro: {}".format(f_macro))
        print("F_micro: {}".format(f_micro))
        print("F_weighted: {}".format(f_weighted))
        print("FC: {}".format(mean(num_selected)))
        print()


def classify_with_fs(r8=True, texts=None, labels=None):
    idx = 0
    vec_helper = TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)
    tdm = vec_helper.fit_transform(texts)
    tdm = tdm.toarray()
    print("# of instances: {}, features: {}".format(str(tdm.shape[0]), str(tdm.shape[1])))
    if r8:
        c_name = "SVM"
        df = pd.DataFrame(columns=["k", c_name])
    else:
        c_name = "LR"
        df = pd.DataFrame(columns=["k", c_name])
    print('-' * 100)
    for k in range(10, tdm.shape[1], 10):
        avg_score = {c_name: []}
        if r8:
            classifier = SVC(kernel='linear', random_state=rs)
        else:
            classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)
        for train_index, test_index in StratifiedKFold(n_splits=5, random_state=rs, shuffle=True).split(tdm, labels):
            x_train, x_test = tdm[train_index], tdm[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            selector = SelectKBest(chi2, k=k)
            x_train_new = selector.fit_transform(x_train, y_train)
            x_test_new = selector.transform(x_test)
            classifier.fit(x_train_new, y_train)
            predictions = classifier.predict(x_test_new)
            f_weighted = f1_score(y_true=y_test, y_pred=predictions, average='weighted')
            avg_score[c_name].append(f_weighted)
        for c_key, res in avg_score.items():
            avg_acc = mean(res)
            print("\t[{} | k: {} | avg_f_weighted: {}]".format(c_key, k, avg_acc))
            df.at[idx, "k"] = k
            df.at[idx, c_key] = avg_acc
        print('-' * 100)
        idx += 1
    print('-' * 100)
    df.to_excel("fs_chi.xlsx")


def filter_based_execution_time(texts=None, labels=None):
    vec_helper = TfidfVectorizer(min_df=mdf, tokenizer=linguistic_tokenizer)
    tdm = vec_helper.fit_transform(texts)
    tdm = tdm.toarray()
    acc_scores, f_time, c_time, f_and_c_time = [], [], [], []
    f_macros, f_micros, f_averages = [], [], []
    idx = 0
    for train_index, test_index in StratifiedKFold(n_folds, random_state=rs, shuffle=True).split(tdm, labels):
        print("Fold: {}".format(idx + 1))
        x_train, x_test = tdm[train_index], tdm[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        fs_time = time.time()
        selector = SelectKBest(chi2, k=490)
        x_train_new = selector.fit_transform(x_train, y_train)
        x_test_new = selector.transform(x_test)
        fe_time = time.time()
        f_time.insert(idx, fe_time - fs_time)
        cs_time = time.time()
        classifier = SVC(kernel='linear', random_state=rs)
        classifier.fit(x_train_new, y_train)
        predictions = classifier.predict(x_test_new)
        ce_time = time.time()
        c_time.insert(idx, ce_time - cs_time)
        f_and_c_time.insert(idx, f_time[idx] + c_time[idx])
        acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
        f_macro = f1_score(y_true=y_test, y_pred=predictions, average='macro')
        f_macros.append(f_macro)
        f_micro = f1_score(y_true=y_test, y_pred=predictions, average='micro')
        f_micros.append(f_micro)
        f_weighted = f1_score(y_true=y_test, y_pred=predictions, average='weighted')
        f_averages.append(f_weighted)
        acc_scores.insert(idx, acc)
        idx += 1
    for k in range(n_folds):
        print("Fold {}: Acc: {} FS Time: {} C Time: {} FS + C Time: {}".format(k, acc_scores[k], f_time[k], c_time[k], f_and_c_time[k]))
    print("Average FS Time: {}".format(mean(f_time)))
    print("Average C Time: {}".format(mean(c_time)))
    print("Average FS + C Time: {}".format(mean(f_and_c_time)))
    print("Average Acc: {}".format(mean(acc_scores)))
    print("# of selected features in each fold: {}".format(selector.k))
    print("Average F_macro: {}".format(mean(f_macros)))
    print("Average F_micro: {}".format(mean(f_micros)))
    print("Average F_weighted: {}".format(mean(f_averages)))


if __name__ == '__main__':
    freeze_support()

    # x, y, targets = get_r8_dataset()
    # stratified_cv_on_benchmark_data(True, x, y)

    # x_train, x_test, y_train, y_test = get_r8_default_train_test_split()
    # holdout_on_benchmark_data(True, x_train, x_test, y_train, y_test)

    # x_train, x_test, y_train, y_test = get_n8_default_train_test_split()
    # holdout_on_benchmark_data(False, x_train, x_test, y_train, y_test)

    x, y, targets = get_subset_of_reuters_dataset()
    # stratified_cv_on_benchmark_data(True, x, y)

    # classify_with_fs(True, x, y)
    filter_based_execution_time(x, y)