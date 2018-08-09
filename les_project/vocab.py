from collections import defaultdict


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
    