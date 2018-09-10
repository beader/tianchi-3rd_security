import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser(description="Encode categorical values to integers")
parser.add_argument('-f', '--data-file', help='CSV file to encode. Read from stdin if ignores.')
parser.add_argument('--chunksize', type=int, default=100000, help='Chunk size for pd.read_csv.')
parser.add_argument('-c', '--columns', help='Categorical columns to encode.')
parser.add_argument('--vocab-file', type=str, required=True, help='Vocabulary file to store raw values and encoded integers.')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Show progress.')
parser.add_argument('--min-int', type=int, default=0, help='Minimum int value the encoder begins with.')
parser.add_argument('-o', '--out-file', help='File to store encoded data. Output to stdout if ignores.')
args = parser.parse_args()


class ColumnsEncoder(object):
    def __init__(self, vocab_file=None, min_int=0):
        self.columns_dict = defaultdict(dict)
        self.vocab_file = vocab_file
        self.min_int = 0
        if vocab_file is not None and os.path.exists(vocab_file):
            self.init_columns_dict()

    def init_columns_dict(self):
        df = pd.read_csv(self.vocab_file, dtype={'column': str, 'value': str, 'encoded': int})
        for _, row in df.iterrows():
            col, val, encoded = row['column'], row['value'], row['encoded']
            self.columns_dict[col][val] = encoded

    def encode_value(self, column, value):
        if value not in self.columns_dict[column]:
            self.columns_dict[column][value] = len(self.columns_dict[column]) + min_int
        return self.columns_dict[column][value]

    def save_to_df(self):
        col_dfs = []
        for column in self.columns_dict.keys():
            col_df = pd.DataFrame([(column, k, v) for k, v in self.columns_dict[column].items()],
                                  columns=['column', 'value', 'encoded'])
            col_dfs.append(col_df)
        df = pd.concat(col_dfs)
        df.to_csv(self.vocab_file, index=False)


def process(input_file, output_file, col_encoder, columns):
    if input_file is None:
        input_file = sys.stdin
    df_chunks = pd.read_csv(input_file, dtype=str, chunksize=args.chunksize)

    if output_file is not None and os.path.exists(output_file):
        os.remove(output_file)
        out_buf = open(output_file, 'a')
    else:
        out_buf = sys.stdout

    if args.verbose:
        pbar = tqdm()

    for i, df in enumerate(df_chunks):
        for col in columns:
            df[col] = df[col].map(lambda x: col_encoder.encode_value(col, x))
        header = True if i == 0 else None
        df.to_csv(out_buf, header=header, index=False)
        if args.verbose:
            pbar.update(len(df))

    col_encoder.save_to_df()
    out_buf.close()


if __name__ == "__main__":
    data_file = args.data_file
    vocab_file = args.vocab_file
    out_file = args.out_file
    min_int = args.min_int
    columns = args.columns.split(',')
    encoder = ColumnsEncoder(vocab_file, min_int)
    process(data_file, out_file, encoder, columns)
