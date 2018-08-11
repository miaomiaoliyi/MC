import json
import numpy as np


class LESDataset(object):
    def __init__(self, max_p_len, max_q_len, vocab, train_file=None, test_file=None):
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.vocab = vocab
        if train_file:
            self.train_set = self._load_dataset(train_file)
        if test_file:
            self.test_set = self._load_dataset(test_file)

    def _load_dataset(self, data_path, train=True):
        """
        加载数据集
        :param data_path:
        :return:
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data_set = json.load(f)
        if train:
            data = []
            for sample in data_set:
                for qa_pairs in sample['questions']:
                    if qa_pairs['answer_spans'][0] == -1:
                        continue
                    data.append({'question': qa_pairs['segmented_question'],
                                 'passage': sample['segmented_article_content'][qa_pairs['most_related_para']],
                                 'answer_span': qa_pairs['answer_spans']})
        return data

    def word_iter(self, set_name):
        if set_name == 'train':
            data_set = self.train_set

        for sample in data_set:
            for question in sample['questions']:
                for word in question['segmented_question']:
                    yield word
                for word in sample['segmented_article_content'][question['most_related_para']]:
                    yield word

    def gen_mini_batches(self, set_name, batch_size, pad_id=0, shuffle=True):
        if set_name == 'train':
            data = self.train_set

        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_data = [data[i] for i in batch_indices]
            yield self._one_mini_batch(batch_data, pad_id)

    def _one_mini_batch(self, batch_data_raw, pad_id):
        batch_data = {'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        for qa_pairs in batch_data_raw:
            batch_data['question_token_ids'].append(self.convert_to_ids(qa_pairs['question'])),
            batch_data['question_length'].append(len(qa_pairs['question']))
            batch_data['passage_token_ids'].append(self.convert_to_ids(qa_pairs['passage']))
            batch_data['passage_length'].append(len(qa_pairs['passage']))
            batch_data['start_id'].append(qa_pairs['answer_span'][0])
            batch_data['end_id'].append(qa_pairs['answer_span'][1])

        batch_data = self._dynamic_padding(batch_data, pad_id)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[:pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[:pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data

    def convert_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab.token2id[token.lower()])
        return ids


