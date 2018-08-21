from collections import defaultdict

import numpy as np


class Vocab(object):

    def __init__(self, filename=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = defaultdict(int)
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = []
        self.initial_tokens.extend([self.pad_token, self.unk_token])

        for token in self.initial_tokens:
            self.add(token)

        if filename:
            self.load_from_file(filename)

    def add(self, token, cnt=True):
        token = token.lower() if self.lower else token

        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        if cnt:
            self.token_cnt[token] += 1

        return idx

    def randomly_init_embeddings(self, embed_dim):
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(len(self.token2id), embed_dim)

        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.token2id[token]] = np.zeros([embed_dim])

    def load_from_file(self, filename):
        pass

    def get_id(self, token):
        token = token.lower if self.lower else token

        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, id):
        try:
            return self.id2token[id]
        except:
            return self.unk_token

    def size(self):
        return len(self.token2id)