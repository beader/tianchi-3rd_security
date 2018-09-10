import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import namedtuple, Counter


parser = argparse.ArgumentParser(description="Convert csv files to tfrecords format.")
parser.add_argument('-f', '--data-file', help='CSV file to encode. Read from stdin if ignores.')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Show progress.')
parser.add_argument('-o', '--out-files', nargs='+',
                    help='''TFRecords files to store examples. If specify multiple [filename:ratio], the data will
                            randomly be splitted into multiple datasets.''')
parser.add_argument('--with-label', action='store_true', default=False, help='Whether the data contains label field.')
parser.add_argument('-z', '--compress', action='store_true', default=True, help='Whether using compression.')
parser.add_argument('-r', '--random-seed', default=0, help='Random seed for spliting dataset.')
args = parser.parse_args()


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


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def example_generator(g, with_label=False):
    for sample in g:
        features = {}
        features['file_id'] = _int64_feature(values=[int(sample[0].file_id)])
        if with_label:
            features['label'] = _int64_feature(values=[int(sample[0].label)])
        df = (pd.DataFrame(sample)[['api', 'index']]
              .astype('int')
              .drop_duplicates()
              .sort_values(['index', 'api']))
        features['api'] = _int64_feature(values=df['api'].values)
        features['index'] = _int64_feature(values=df['index'].values)
        features['seq_len'] = _int64_feature(values=[df['index'].max() + 1])
        features['api_cnt'] = _int64_feature(values=[len(df)])
        example = tf.train.Example(features=tf.train.Features(feature=features))
        yield example


class RandomSplitTFRecordWriter(object):
    def __init__(self, split_ratio=None):
        assert isinstance(split_ratio, dict)
        ext = '.tfrecords.gz' if args.compress else '.tfrecords'
        np.random.seed(args.random_seed)
        self.counter = Counter()
        if args.compress:
            tfr_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        else:
            tfr_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        self.filenames = [fname + ext for fname in split_ratio.keys()]
        self.probabilities = list(split_ratio.values())
        self.writers = [tf.python_io.TFRecordWriter(fname, tfr_options) for fname in self.filenames]

    def write(self, record):
        i = np.random.choice(range(len(self.writers)), 1, p=self.probabilities)[0]
        self.counter[self.filenames[i]] += 1
        self.writers[i].write(record)

    def close(self):
        for writer in self.writers:
            writer.close()

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        self.close()
        for fname, cnt in self.counter.items():
            print('{} samples written to {}'.format(cnt, fname))


def process():
    out_files = {}
    for out_file in args.out_files:
        parsed = out_file.split(':')
        fname = parsed[0]
        if len(parsed) == 2:
            out_files[fname] = float(parsed[1])
        else:
            out_files[fname] = 1.0
    assert sum(out_files.values()) == 1.0

    g = line_generator(args.data_file)
    g = row_generator(g)
    g = group_generator(g, 'file_id')
    g = example_generator(g, with_label=args.with_label)
    if args.verbose:
        pbar = tqdm()

    with RandomSplitTFRecordWriter(out_files) as writer:
        for example in g:
            writer.write(example.SerializeToString())
            if args.verbose:
                pbar.update(1)


if __name__ == "__main__":
    process()
