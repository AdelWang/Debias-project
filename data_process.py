import math
import jieba
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
import pandas as pd
import nltk
import nltk.stem as ns
from collections import defaultdict
import json

data_path = 'D:/data_mining/NLP/project/overlap/dataset/MNLI/MNLI/train_overlap.txt'
model_name_or_path = 'D:/data_mining/NLP/project/overlap/PLM/bert_base'
lemmatizer = ns.WordNetLemmatizer()
tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
def text_to_csv(text_path, csv_path, hard = False):
    f = open(text_path, 'r', encoding='utf-8')
    res = []
    for line in f:
        data = line.strip('\n').split('\t')
        column_num = len(data)
        res.append(data)
    f.close()
    colunms_total = ['sentence1', 'sentence2', 'label', 'overlap','hard']
    true_column = colunms_total[:column_num]
    dataset = pd.DataFrame(columns=true_column, index=None, data=list(res))
    fh = open(csv_path, 'a', encoding='utf-8-sig',newline='')
    dataset.to_csv(fh, index=False, encoding='utf-8-sig')
    fh.close()
def read_dict(path):
    dict_ = defaultdict(int)
    f = open(path,'r',encoding='utf-8')
    for line in f:
        data = line.strip()
        dict_[data] = 1
    f.close()
    return dict_

stop_word = "D:/data_mining/NLP/project/overlap/dataset/stopwords/stopwords-master/stopwords-master/self_defined.txt"
stop_word_dict = read_dict(stop_word)

def result_to_csv(text_path, csv_path):
    f = open(text_path, 'r', encoding='utf-8')
    res = []
    for line in f:
        line = line.strip('\n')
        data = line.split('\t')
        if not data[0].isdigit():
            continue
        column_num = len(data)
        res.append((str(data[0]),str(data[1])))
    f.close()
    colunms_total = ["pairID", "gold_label"]
    true_column = colunms_total[:column_num]
    dataset = pd.DataFrame(columns=true_column, index=None, data=list(res))
    fh = open(csv_path, 'a', encoding='utf-8-sig',newline='')
    dataset.to_csv(fh, index=False, encoding='utf-8-sig')
    fh.close()

def overlap_calculate_chinese(text_path, result_path):
    f = open(text_path, 'r', encoding='utf-8')
    for index, line in enumerate(f):
        if index % 500 == 0:
            print("process: ",index)
        data = line.rstrip().split('\t')
        lap_total = 0
        len_same = 0
        dict1 = {}
        dict2 = {}
        sen1 = jieba.lcut(data[0])
        sen2 = jieba.lcut(data[1])
        for word1 in sen1:
            if word1 not in dict1:
                dict1[word1] = 1
            else:
                dict1[word1] += 1
        for word2 in sen2:
            if word2 not in dict2:
                dict2[word2] = 1
            else:
                dict2[word2] += 1

        for dict_word in dict1:
            if dict_word not in dict2:
                pass
            else:
                l_lap = min(dict1[dict_word], dict2[dict_word])
                len_same += l_lap * len(dict_word)
        p = 2 * len_same / (len(data[0]) + len(data[1]))
        if data[-1] == 'neutral':
            label = 2
        elif data[-1] == 'entailment':
            label = 0
        elif data[-1] == 'contradiction':
            label = 1
        else:
            label = data[-1]
        f1 = open(result_path, 'a', encoding='utf-8')
        f1.writelines([str(data[0]), '\t', str(data[1]), '\t', str(label), '\t', str(p), '\n'])
        f1.close()
    f.close()


def lemme(word, lemmatizer, use_stop_words=False):
    if use_stop_words:
        if word in stop_word_dict:
            return ""
    lemm = lemmatizer.lemmatize(word, pos='n')
    lemm_final = lemmatizer.lemmatize(lemm, pos='v')
    return lemm_final


def label_map(label):
    label_table = {
        # NLI task
        "entailment": 1,
        "neutral": 2,
        "contradiction": 0,
        "non-entailment": 0,
        "not_entailment": 0,
        "1":1,
        "0":0
    }
    return label_table[label]


def hard_class_map(hard_type):
    hard_type_table = {
        'lexical_overlap': 0,
        'subsequence': 1,
        'constituent': 2,
    }
    return hard_type_table[hard_type]


def overlap_calculate(text_path, result_path, look_up, use_stop_words=False, heading=False, hard=False):
    f = open(text_path, 'r', encoding='utf-8')
    f1 = open(result_path, 'a', encoding='utf-8')
    index_sen1, index_sen2, index_label = look_up
    for index, line in enumerate(f):
        if not heading:
            if index < 1:
                continue
        data = line.rstrip().split('\t')
        if data[index_sen1] == 'n/a':
            data[index_sen1] = 'no sentence'
        elif data[index_sen2] == 'n/a':
            data[index_sen2] = 'no sentence'
        lap_total = 0
        dict1 = defaultdict(int)
        dict2 = defaultdict(int)
        sen1 = tokenizer.tokenize(data[index_sen1])
        sen2 = tokenizer.tokenize(data[index_sen2])
        for word1 in sen1:
            lemme_word1 = lemme(word1,lemmatizer,use_stop_words)
            dict1[lemme_word1] += 1
        for word2 in sen2:
            lemme_word2 = lemme(word2, lemmatizer,use_stop_words)
            dict2[lemme_word2] += 1
        for dict_word in dict1:
            if dict_word not in dict2:
                pass
            else:
                l_lap = min(dict1[dict_word], dict2[dict_word])
                lap_total += l_lap
        p = 2 * lap_total / (len(sen1) + len(sen2))
        if not data[index_label].isdigit():
            label = label_map(data[index_label])
        else:
            label = data[index_label]
        if hard:
            hard_label = hard_class_map(data[8])
            f1.writelines(
                [str(data[index_sen1]), '\t', str(data[index_sen2]), '\t', str(label), '\t', str(p), '\t', str(hard_label), '\n'])
        else:
            f1.writelines([str(data[index_sen1]), '\t', str(data[index_sen2]), '\t', str(label), '\t', str(p), '\n'])
    f1.close()
    f.close()

def processor(text_path, result_path, is_chinese=False, is_test=False):
    f = open(text_path,'r',encoding='utf-8')
    for index, line in enumerate(f):
        if not is_chinese and index < 1:
            continue
        elif index % 500 == 0:
            print(index)
        data = line.rstrip().split('\t')
        if not is_test:
            if data[-1] == 'neutral':
                label = 2
            elif data[-1] == 'entailment':
                label = 0
            elif data[-1] == 'contradiction':
                label = 1
            elif data[-1] == 'not_entailment':
                label = 1
            else: label = None
            f1 = open(result_path, 'a', encoding='utf-8')
            if label is not None:
                f1.writelines([str(data[1]), '\t', str(data[2]), '\t', str(label),'\n'])
            else:
                f1.writelines([str(data[1]), '\t', str(data[2]), '\t', str(data[-1]), '\n'])
            f1.close()
        else:
            f1 = open(result_path, 'a', encoding='utf-8')
            f1.writelines([str(data[5]), '\t', str(data[6]),'\t', str(data[8]), '\n'])
            f1.close()
    f.close()
def clean_paws(path, path_result):
    stop_word = ["b",'"',"'"]
    stop_word_dict = defaultdict(int)
    for word in stop_word:
        id = tokenizer.encode(word,add_special_tokens=False)
        stop_word_dict[id[0]] = 1
    f = open(path, 'r',encoding='utf-8')
    f_clean = open(path_result,'a',encoding='utf-8')
    for index, line in enumerate(f):
        data = line.rstrip().split('\t')
        sen1 = data[0]
        sen2 = data[1]
        token_sen1 = tokenizer.encode(sen1,add_special_tokens=False)
        token_sen2 = tokenizer.encode(sen2,add_special_tokens=False)
        clean_1 = []
        clean_2 = []
        for token in token_sen1:
            if token not in stop_word_dict:
                clean_1.append(token)
        for token in token_sen2:
            if token not in stop_word_dict:
                clean_2.append(token)
        clean_sen1 = tokenizer.decode(clean_1)
        clean_sen2 = tokenizer.decode(clean_2)
        f_clean.writelines([clean_sen1,'\t',clean_sen2,'\t',data[-2],'\t',data[-1],'\n'])
    f.close()
    f_clean.close()
    return 

def na_counter(text_path, heading=False, hard=False):
    f = open(text_path, 'r', encoding='utf-8')
    #f1 = open(result_path, 'a', encoding='utf-8')
    num_na = 0
    num_0 = 0
    for index, line in enumerate(f):
        if not heading:
            if index < 1:
                continue
        data = line.rstrip().split('\t')
        if data[5] == 'n/a' or 'no' in data[5]:
            num_na += 1
            if data[0] == "non-entailment":
                num_0 += 1
        elif data[6] == 'n/a' or 'no' in data[6]:
            num_na += 1
            if data[0] == "non-entailment":
                num_0 += 1
    print(num_na, num_0)
    f.close()
    return
def caculate_ave(path):
    record_overlap = defaultdict(float)
    record_nums = defaultdict(int)
    f = open(path,'r',encoding='utf-8')
    for index, line in enumerate(f):
        data = line.rstrip().split('\t')
        record_overlap[data[-2]] += float(data[-1])
        record_nums[data[-2]] += 1
    ave_1 = record_overlap["1"] / record_nums["1"]
    ave_2 = (record_overlap["0"] + record_overlap["2"]) / (record_nums["0"] + record_nums["2"])
    return ave_1, ave_2


if __name__ == '__main__':
    path1 = "D:/data_mining/NLP/project/overlap/dataset/HANS/heuristics_train_set.txt"
    data_look_up = {
        "qqp":[3,4,-1],
        "mnli":[8,9,-1],
        "MNLI":[8,9,-1],
        "heuristics":[5,6,0],
        "mrpc":[3,4,0],
        "rte":[1,2,-1]
    }
    path_split = path1.split(".")
    path2 = path_split[0] + "_map_overlap.txt"
    path3 = path_split[0] + "_map_overlap.csv"
    for key in data_look_up.keys():
        if key in path1:
            task_name = key
    look_up = data_look_up[task_name]
    overlap_calculate(path1,path2,look_up)
    text_to_csv(path2,path3)
    '''
    path_split = path1.split(".")
    path2 = path_split[0] + "_map.txt"
    path3 = path_split[0] + "_map.csv"
    f = open(path1,'r',encoding='utf-8')
    f_map = open(path2,'a',encoding='utf-8')
    for index,line in enumerate(f):
        data = line.rstrip().split('\t')
        if data[-2] == '0':
            label = 1
        elif data[-2] == '1':
            label = 0
        else:
            label = 2
        f_map.writelines([data[0],'\t',data[1],'\t',str(label),'\t',data[-1],'\n'])
    f.close()
    f_map.close()
    text_to_csv(path2, path3)
    '''





