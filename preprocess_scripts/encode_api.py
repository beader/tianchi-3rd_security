import os
import argparse
import pandas as pd
from zipfile import ZipFile
from tqdm import tqdm
from collections import namedtuple


parser = argparse.ArgumentParser(description="Encode api to numbers")
parser.add_argument('--data-dir', default='./data', help='data dir that contains train.csv and test.csv')
args = parser.parse_args()


def zipfile_line_generator(filename):
    with ZipFile(filename) as zf:
        with zf.open(zf.namelist()[0]) as f:
            for line in f:
                yield line.decode('utf-8').strip()


def txt_line_generator(filename):
    with open(filename, encoding='utf-8') as f:
        for line in f:
            yield line.strip()


def row_generator(g):
    fields = next(g).strip().split(',')
    Row = namedtuple('Row', fields)
    for line in g:
        yield Row(*line.split(','))


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


class LabelEncoder(object):
    def __init__(self):
        self.val_dict = {}

    def encode_value(self, val):
        if val not in self.val_dict:
            self.val_dict[val] = str(len(self.val_dict))
        return self.val_dict[val]


def process(input_file, output_file, label_encoder):
    print('processing {} and saving to {}'.format(input_file, output_file))
    _, ext = os.path.splitext(input_file)
    if ext == '.zip':
        data = zipfile_line_generator(input_file)
    else:
        data = txt_line_generator(input_file)
    data = row_generator(data)
    data = group_generator(data, ['file_id'])
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'a') as f:
        for i, t in tqdm(enumerate(data)):
            df = pd.DataFrame(t)
            df['index'] = df['index'].astype('int')
            df['api'] = [label_encoder.encode_value(api) for api in df['api']]
            header = True if i == 0 else None
            df.sort_values(['index', 'tid']).to_csv(f, header=header, index=False)


if __name__ == "__main__":
    data_dir = args.data_dir
    le = LabelEncoder()

    for data_file in ['train.csv', 'test.csv']:
        fname, _ = os.path.splitext(data_file)
        data_file = os.path.join(data_dir, data_file)
        output_file = os.path.join(data_dir, '{}_encoded.csv'.format(fname))
        if not os.path.exists(data_file):
            data_file = os.path.join(data_dir, '3rd_security_{}.zip'.format(fname))
        process(data_file, output_file, le)

    api_mapping_file = os.path.join(data_dir, 'api_mapping.csv')
    api_mapping_df = pd.DataFrame([(k, int(v)) for k, v in le.val_dict.items()], columns=['api', 'encoded'])
    api_mapping_df = api_mapping_df.sort_values('encoded').reset_index(drop=True)
    api_mapping_df.to_csv(api_mapping_file, index=False)
