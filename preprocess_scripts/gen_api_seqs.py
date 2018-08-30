import os
import argparse
import pandas as pd
from tqdm import tqdm
from collections import namedtuple

parser = argparse.ArgumentParser(description="Generate api sequences and pickle it as a pandas dataframe")
parser.add_argument('--data-dir', default='./data',
                    help='data dir that contains train_encoded.csv and test_encoded.csv')
args = parser.parse_args()


def row_generator(filename):
    with open(filename, encoding='utf-8') as f:
        fields = f.readline().strip().split(',')
        Row = namedtuple('Row', fields)
        for line in f:
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


def process(input_file, output_file):
    print('processing {} and saving to {}'.format(input_file, output_file))
    data = row_generator(input_file)
    data = group_generator(data, 'file_id')
    file_ids = []
    file_api_seqs = []
    file_labels = []
    for group_data in tqdm(data):
        file_id = int(group_data[0].file_id)
        if hasattr(group_data[0], 'label'):
            file_labels.append(getattr(group_data[0], 'label'))
        api_seqs = []
        for index_data in group_generator(group_data, 'index'):
            apis = list(set(int(d.api) for d in index_data))
            api_seqs.append(apis)
        file_ids.append(file_id)
        file_api_seqs.append(api_seqs)
    df = pd.DataFrame({'file_id': file_ids, 'api_seq': file_api_seqs})
    if len(file_labels) > 0:
        df['label'] = file_labels
    df.to_pickle(output_file)


if __name__ == "__main__":
    data_dir = args.data_dir

    for data_file in ['train_encoded.csv', 'test_encoded.csv']:
        fname, _ = os.path.splitext(data_file)
        data_file = os.path.join(data_dir, data_file)
        output_file = os.path.join(data_dir, '{}_api_seqs.pkl'.format(fname))
        print(data_file, output_file)
        process(data_file, output_file)
