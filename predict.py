import pandas as pd
import argparse
from data_utils import APISequence, train_test_split
from keras.models import load_model

parser = argparse.ArgumentParser(description="Train classifier")
parser.add_argument('--test-pkl-file', default='./data/test_encoded_api_seqs.pkl',
                    help='pickled test data')
parser.add_argument('--max-seq-len', type=int, default=5001,
                    help='max sequence length')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--model-file', help='model file path')
parser.add_argument('--submission-file', default='submission.csv', help='file to save predictions')
args = parser.parse_args()


NUM_WORDS = 311
NUM_CLASSES = 6

if __name__ == "__main__":
    df = pd.read_pickle(args.test_pkl_file)
    api_seq = APISequence(df['api_seq'].values, args.max_seq_len, NUM_WORDS)
    model = load_model(args.model_file)
    preds = model.predict(api_seq, verbose=True, batch_size=args.batch_size)
    submit_df = pd.DataFrame(preds, columns=['prob' + str(i) for i in range(NUM_CLASSES)])
    submit_df = pd.concat([df[['file_id']], submit_df], axis=1)
    submit_df.to_csv(args.submission_file, index=False)
