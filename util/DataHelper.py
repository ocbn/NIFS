import os
import pandas as pd
import string

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')


def format_cyberbully_data():
    path = desktop + "/dataset"
    directories = {
        "EV": "ErkekHakaretVar",
        "EY": "ErkekHakaretYok",
        "BY": "BayanHakaretYok",
        "BV": "BayanHakaretVar"}
    gender = []
    cyberbully = []
    emoticons = {
        "...": "XA",
        ":)": "XB",
        ":-)": "XC",
        "<3": "XD",
        ":(": "XE",
        ":P": "XF",
        ":p": "XG",
        "8)": "XH",
        ":*": "XI",
        ">:(": "XJ",
        ":D": "XL",
        ":O": "XM",
        ":|": "XN",
        "O:)": "XO",
        ":@": "XP",
        ";)": "XR",
        ";(": "XS",
        ";*": "XT",
        "!": "XO"
    }
    punctuations = '''!()-[]{};:'",<>./?@#$%^&*_~'''
    for code, d in directories.items():
        sub_directory = path + "/" + d
        files = os.listdir(sub_directory)
        for f in files:
            file_path = sub_directory + "/" + f
            f = open(file_path, "r")
            content = " ".join(" ".join(f.readlines()).strip().replace(".", " ").split())
            for k, v in emoticons.items():
                content = content.replace(k, " " + v + " ")
            content = "".join(i for i in content if not i.isdigit())
            temp = ""
            for char in content:
                if char not in punctuations:
                    temp = temp + char
                else:
                    temp = temp + " "
            content = " ".join(temp.split()).lower()
            if code in ["EV", "EY"]:  # male
                gender.append({"text": content, "gender": 1})
            else:  # female
                gender.append({"text": content, "gender": 0})

            if code in ["EV", "BV"]:
                cyberbully.append({"text": content, "cyberbully": 1})
            else:
                cyberbully.append({"text": content, "cyberbully": 0})
    print("# of samples in gender: {}".format(len(gender)))
    print("# of samples in cyberbully: {}".format(len(cyberbully)))

    # df_gender = pd.DataFrame(data=gender)
    # df_cyberbully = pd.DataFrame(data=cyberbully)
    # df_gender.to_excel("cyberbully_gender.xlsx", index=False)
    # df_cyberbully.to_excel("cyberbully.xlsx", index=False)


def format_ttc_3600_data():
    path = desktop + "/ttc"
    directories = {
        "1": "ekonomi",
        "2": "kultursanat",
        "3": "saglik",
        "4": "siyaset",
        "5": "spor",
        "6": "teknoloji"
    }
    data = []
    for code, d in directories.items():
        sub_directory = path + "/" + d
        files = os.listdir(sub_directory)
        for f in files:
            file_path = sub_directory + "/" + f
            f = open(file_path, "r", encoding="utf-8")
            content = " ".join(" ".join(f.readlines()).strip().rstrip().replace(".", " ").split())
            content = "".join(i for i in content if not i.isdigit())
            content = " ".join(content.split()).lower().strip().rstrip()
            data.append({"text": content, "label": code})
    print("# of samples in gender: {}".format(len(data)))

    # df_gender = pd.DataFrame(data=gender)
    # df_cyberbully = pd.DataFrame(data=cyberbully)
    # df_gender.to_excel("cyberbully_gender.xlsx", index=False)
    # df_cyberbully.to_excel("cyberbully.xlsx", index=False)


path = desktop + "/ttc.csv"
data = pd.read_csv(path)
texts = list(data["text"])
labels = list(data["category"])
cats = {"dunya": 1, "ekonomi": 2, "kultur": 3, "saglik": 4, "siyaset": 5, "spor": 6, "teknoloji": 7}
punctuations = '''!()-[]{};:'",<>./?@#$%^&*_~'''
p_data = []
instance_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
for i in range(len(texts)):
    text = texts[i]
    text = str(text).strip().rstrip()
    text = "".join(i for i in text if not i.isdigit())
    text = " ".join(text.split()).lower().strip().rstrip()
    # temp = ""
    # for char in text:
    #     if char not in punctuations:
    #         temp = temp + char
    #     else:
    #         temp = temp + " "
    # text = " ".join(temp.split()).lower().strip().rstrip()
    label = cats[str(labels[i]).lower().strip().rstrip()]
    if instance_counts[label] <= 49:
        p_data.append({"text": str(text), "label": label})
        instance_counts[label] += 1
df_data = pd.DataFrame(data=p_data)
df_data.to_excel("ttc_digits_removed.xlsx", index=False, encoding="utf-8")
