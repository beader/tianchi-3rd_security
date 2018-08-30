import numpy as np
from operator import itemgetter
from keras.utils import to_categorical
from keras.utils.data_utils import Sequence
from keras.preprocessing.sequence import pad_sequences


class APISequence(object):
    def __init__(self, api_seqs, seq_len, num_apis):
        self.api_seqs = api_seqs
        self.seq_len = seq_len
        self.num_apis = num_apis

    def __len__(self):
        return len(self.api_seqs)

    @property
    def shape(self):
        return len(self), self.seq_len, self.num_apis

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def _random_trim_seq(seq, trim_len):
        trim_indices = np.random.choice(np.arange(len(seq)), trim_len, replace=False)
        return itemgetter(*trim_indices)(seq)

    def _encode_seq(self, seq):
        if len(seq) > self.seq_len:
            seq = self._random_trim_seq(seq, self.seq_len)
        return np.array([to_categorical(api, self.num_apis).sum(axis=0) for api in seq])

    def _encode_seqs(self, seqs):
        return pad_sequences([self._encode_seq(seq) for seq in seqs],
                             maxlen=self.seq_len,
                             padding='post',
                             value=[0] * self.num_apis)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            api_seqs = [self.api_seqs[indices]]
        else:
            api_seqs = self.api_seqs[indices]
        api_seqs = self._encode_seqs(api_seqs)
        return api_seqs


class RandomReweightedDataSequence(Sequence):
    def __init__(self, X, y, batch_size=64, reweight_ratios=None, global_indices=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.global_indices = np.arange(len(y)) if global_indices is None else global_indices
        self.reweight_ratios = reweight_ratios if isinstance(reweight_ratios, dict) else dict()
        self.stratified_indices = self._get_stratified_indices()
        self.on_epoch_end()

    @property
    def num_samples(self):
        return len(self.sample_ids)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def _gen_sample_ids(self):
        sample_ids = []
        for label, indices in self.stratified_indices.items():
            ratio = self.reweight_ratios.get(label, 1)
            sample_cnt = int(ratio * len(indices))
            sample_ids.append(np.random.choice(indices, sample_cnt, replace=False))
        sample_ids = np.concatenate(sample_ids)
        np.random.shuffle(sample_ids)
        return sample_ids

    def _get_stratified_indices(self):
        stratified_indices = dict()
        y = self.y[self.global_indices]
        uniq_labels = np.unique(y)
        for label in uniq_labels:
            stratified_indices[label] = self.global_indices[np.where(y == label)[0]]
        return stratified_indices

    def on_epoch_end(self):
        self.sample_ids = self._gen_sample_ids()

    def __getitem__(self, batch_id):
        batch_data_index = self.sample_ids[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
        batch_x = self.X[batch_data_index]
        batch_y = self.y[batch_data_index]
        return batch_x, batch_y


def train_test_split(X, y, batch_size=64, validation_split=0.1, reweight_ratios=dict()):
    num_samples = len(y)
    global_indices = np.arange(num_samples)
    np.random.shuffle(global_indices)

    num_val_samples = int(num_samples * validation_split)

    train_sample_indices = global_indices[num_val_samples:].copy()
    val_sample_indices = global_indices[:num_val_samples].copy()

    train_data = RandomReweightedDataSequence(X, y, batch_size=batch_size, reweight_ratios=reweight_ratios,
                                              global_indices=train_sample_indices)
    val_data = RandomReweightedDataSequence(X, y, batch_size=batch_size, reweight_ratios=reweight_ratios,
                                            global_indices=val_sample_indices)
    return train_data, val_data
