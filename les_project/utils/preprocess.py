import os
import re
import json
from collections import Counter

import jieba
import numpy as np
from tqdm import tqdm


def precision_recall_f1(prediction, ground_truth):
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
    score = metric_fn(prediction, ground_truth)
    return score


def load_data_set(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_set = json.loads(f.read())
    return data_set


# 找到最相关的段落和在段落中的位置
def find_fake_answer(sample):
    for a_idx, answer_token in enumerate(sample['questions']):
        most_related_para = -1
        most_related_para_len = 999999
        max_related_score = 0
        #         print('a_idx=',a_idx, 'answer_token=',answer_token)
        for p_idx, para_tokens in enumerate(sample['segmented_article_content']):
            related_score = metric_max_over_ground_truths(recall,
                                                          para_tokens,
                                                          answer_token['segmented_answer'])
            #             print('p_idx=',p_idx,'related_score=',related_score)
            if related_score > max_related_score \
                    or (related_score == max_related_score
                        and len(para_tokens) < most_related_para_len):
                most_related_para = p_idx
                most_related_para_len = len(para_tokens)
                max_related_score = related_score
        sample['questions'][a_idx]['most_related_para'] = most_related_para
        most_related_para_tokens = sample['segmented_article_content'][most_related_para]

        answer_tokens = set(answer_token['segmented_answer'])
        best_match_score = 0
        best_match_span = [-1, -1]
        best_fake_answer = None

        for start_tidx in range(len(most_related_para_tokens)):
            if most_related_para_tokens[start_tidx] not in answer_tokens:
                continue
            for end_tidx in range(len(most_related_para_tokens) - 1, start_tidx - 1, -1):
                span_tokens = most_related_para_tokens[start_tidx: end_tidx + 1]
                match_score = metric_max_over_ground_truths(f1_score, span_tokens,
                                                            answer_token['segmented_answer'])
                if match_score == 0:
                    break
                if match_score > best_match_score:
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_fake_answer = ''.join(span_tokens)
        sample['questions'][a_idx]['answer_spans'] = best_match_span
        sample['questions'][a_idx]['fake_answers'] = best_fake_answer
        sample['questions'][a_idx]['match_scores'] = best_match_score
    return sample


def clean_data(sample):
    # 文章内容和标题分段->分词：将标题插入到分段后的首位置
    sample['segmented_article_title'] = \
        list(jieba.cut(''.join(re.split(r'\u3000+|\s+|\t+', sample['article_title'].strip()))))

    sample_splited_para = re.split(r'\u3000+|\s+|\t+', sample['article_content'].strip())
    if len(sample_splited_para) == 1 and len(sample_splited_para[0]) > 200:
        sample_splited_para = re.split(r'\。', sample['article_content'].strip())
    sample_splited_list = []
    for para in sample_splited_para:
        sample_splited_list.append(list(jieba.cut(para.strip(), cut_all=False)))
    sample_splited_list.insert(0, sample['segmented_article_title'])

    sample['segmented_article_content'] = sample_splited_list

    # 问题和答案分词处理
    for i, question in enumerate(sample['questions']):
        sample['questions'][i]['segmented_question'] = \
            list(jieba.cut(''.join(question['question'].strip().split('\u3000+|\s+|\t+'))))
        sample['questions'][i]['segmented_answer'] = \
            list(jieba.cut(''.join(question['answer'].strip().split('\u3000+|\s+|\t+'))))
    return sample


def run_preprocess(file_path, start=0, end=-1):
    def save(data, i):
        with open('../../data/preprocessed_%d.json' % i, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    data_set = load_data_set(file_path)
    data_preprocessed = []
    for i, sample in enumerate(data_set[start: end + 1]):
        if i % 100 == 0 and i != 0:
            print(i + start)
            save(data_preprocessed, (i + start) / 100)
            data_preprocessed = []

        sample_preprocessed = find_fake_answer(clean_data(sample))
        data_preprocessed.append(sample_preprocessed)


def store_prerpocess_data():
    preprocessed_data = []
    for i in range(1, 201):
        with open('./data/preprocessed_%d.json' % i, 'r', encoding='utf-8') as f:
            d = json.load(f)
        preprocessed_data.extend(d)
    with open('./data/preprocessed.json', 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f)


def save_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def qc(data_preprocessed):
    """
    原始数据集中有重复的文章，根据标题去重
    :return:
    """
    title_set = set()
    data_qc = []
    for sample in data_preprocessed:
        title = sample['article_title']
        if title in title_set:
            continue
        else:
            title_set.add(title)
            data_qc.append(sample)
    save_data(data_qc, os.path.join('F:\\jupyter_file\\MC\\data', 'preprocessed.json'))


def train_test_split(data_path, train_percent=0.9):
    """
    切分训练集，测试集
    :param data_path:
    :param train_percent:
    :return:
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    index = np.arange(len(dataset))
    np.random.shuffle(index)

    train_size = int(len(dataset) * train_percent)
    train_index = index[:train_size]
    test_index = index[train_size:]
    train_set, test_set = [], []
    for index in train_index:
        train_set.append(dataset[index])
    for index in test_index:
        test_set.append(dataset[index])

    save_data(train_set, os.path.join('F:\\jupyter_file\\MC\\data', 'trainset.json'))
    save_data(test_set, os.path.join('F:\\jupyter_file\\MC\\data', 'testset.json'))


if __name__ == '__main__':
    # run_preprocess('../../data/question.json', start=19900, end=20000)
    train_test_split(os.path.join('F:\\jupyter_file\\MC\\data', 'preprocessed_qc.json'))
