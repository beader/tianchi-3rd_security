import pandas as pd
from scipy.sparse import coo_matrix, vstack
from collections import Counter, namedtuple
from functools import reduce
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


def _word_ngrams(tokens, ngram_range, stop_words=None):
    """Turn tokens into a sequence of n-grams after stop words filtering
    copy from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L148
    """
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        if min_n == 1:
            # no need to do any slicing for unigrams
            # just iterate through the original tokens
            tokens = list(original_tokens)
            min_n += 1
        else:
            tokens = []

        n_original_tokens = len(original_tokens)

        # bind method outside of loop to reduce overhead
        tokens_append = tokens.append
        space_join = " ".join

        for n in range(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens_append(space_join(original_tokens[i: i + n]))

    return tokens

def line_generator(filename):
    if filename is not None:
        f = open(filename, encoding='utf-8')
    else:
        f = sys.stdin
    for line in f:
        yield line
    f.close()


def row_generator(g):
    fields = next(g).strip().split(',')
    Row = namedtuple('Row', fields)
    for line in g:
        yield Row(*line.strip().split(','))


def group_generator(g, group_keys):
    buffer = []
    last_gid = None
    if not isinstance(group_keys, list):
        group_keys = [group_keys]
    for row in g:
        cur_gid = tuple(getattr(row, k) for k in group_keys)
        if last_gid is None:
            last_gid = cur_gid
        if cur_gid != last_gid:
            last_gid = cur_gid
            yield buffer
            buffer = []
        buffer.append(row)
    yield buffer

def ngram_generator(g, ngram_range):
    ThreadSample = namedtuple('ThreadSample', ['file_id', 'ngrams', 'label'])
    for group in g:
        ng_cnt = Counter()
        file_id = group[0].file_id
        label = getattr(group[0], 'label', -1)
        apis = [row.api for row in group]
        for gram in _word_ngrams(apis, ngram_range):
            ng_cnt[gram] += 1
        yield ThreadSample(file_id, ng_cnt, label)

def file_sample_generator(g):
    FileSample = namedtuple('FileSample', ['file_id', 'ngrams', 'label'])
    for group in g:
        file_id = group[0].file_id
        label = getattr(group[0], 'label', -1)
        ngrams = reduce(Counter.__add__, (t.ngrams for t in group))
        yield FileSample(file_id, ngrams, label)

class ValueEncoder(object):
    def __init__(self):
        self.val_dict = {}

    def encode(self, val):
        if val not in self.val_dict:
            self.val_dict[val] = len(self.val_dict)
        return self.val_dict[val]

def extract_bow_features_from_file(path, value_encoder, ngram_range=(1,), total=None):
    g = line_generator(path)
    g = row_generator(g)
    g = group_generator(g, ['file_id', 'tid'])
    g = ngram_generator(g, ngram_range=(1, 3))
    g = group_generator(g, 'file_id')
    g = file_sample_generator(g)
    rows, cols, values = [], [], []
    labels = []
    for file_sample in tqdm(g, total=total, ncols=80):
        file_id = int(file_sample.file_id)
        label = int(file_sample.label)
        for ngram, cnt in file_sample.ngrams.items():
            rows.append(file_id)
            cols.append(value_encoder.encode(ngram))
            values.append(cnt)
        labels.append(label)
    bow_feats = coo_matrix((values, (rows, cols)))
    return bow_feats, labels


if __name__ == '__main__':
    train_file = './data/train.csv'
    test_file = './data/test.csv'

    ve = ValueEncoder()
    print('Extracting (1,3)-grams from train file ...')
    tr_bow_feats, tr_labels = extract_bow_features_from_file(train_file, ve,
                                                             ngram_range=(1, 3), total=116624)
    print('Extracting (1,3)-grams from test file ...')
    te_bow_feats, _ = extract_bow_features_from_file(test_file, ve,
                                                     ngram_range=(1, 3), total=53093)
    print('Calculating TFIDF Value ...')
    num_feats = te_bow_feats.shape[1]
    tr_bow_feats.resize((tr_bow_feats.shape[0], num_feats))

    tfidf_tsfm = TfidfTransformer()
    tfidf_tsfm.fit(vstack((tr_bow_feats, te_bow_feats)))
    tr_bow_feats = tfidf_tsfm.transform(tr_bow_feats)
    te_bow_feats = tfidf_tsfm.transform(te_bow_feats)
    rfc = RandomForestClassifier(n_estimators=50, n_jobs=-1)

    print('Cross validation')
    cv_res = cross_validate(rfc, tr_bow_feats, tr_labels, scoring='neg_log_loss', return_train_score=True)
    print(cv_res)

    print('Training RandomForestClassifier')
    rfc.fit(tr_bow_feats, tr_labels)

    print('Making submission')
    preds = rfc.predict_proba(te_bow_feats)
    submit_df = pd.concat([
        pd.DataFrame({'file_id': range(len(preds))}),
        pd.DataFrame(preds, columns=['prob' + str(i) for i in range(6)])
    ], axis=1)

    print('Output to randomforest_baseline.csv')
    submit_df.to_csv('randomforest_baseline.csv', index=False)

