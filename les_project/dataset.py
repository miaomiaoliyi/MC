import json


class LESDataset(object):
    def __init__(self, max_p_len, max_q_len, train_file=None, dev_file=None, test_file=None):
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []


    def _load_dataset(self, data_path):
        """
        加载数据集
        :param data_path:
        :return:
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data_set = json.load(f)
        return data_set


    def word_iter(self, set_name):
        if set_name == 'train':
            data_set = self.train_set

        for sample in data_set:
            for question in sample['questions']:
                for word in question['segmented_question']:
                    yield word
                for word in sample['segmented_article_content'][question['most_related_para']]:
                    yield word

