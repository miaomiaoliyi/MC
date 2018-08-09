import pickle

from dataset import LESDataset
from vocab import Vocab


def prepare():
    les_data = LESDataset(max_p_len=300, max_q_len=60, train_file='..//data//preprocessed.json')

    vocab =Vocab(lower=True)
    for word in les_data.word_iter('train'):
        vocab.add(word)

    with open('..//data//vocab.data', 'wb') as f:
        pickle.dump(vocab, f)


if __name__ == '__main__':
    prepare()