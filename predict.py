import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from data_utils import read_tfrecord_dataset, SampleDecoder, api_one_hot, count_tfrecord

parser = argparse.ArgumentParser(description="Train classifier")
parser.add_argument('--test-file', required=True, help='TFRecord file for tesing.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--model-file', required=True, help='model file path')
parser.add_argument('--submission-file', required=True, default='submission.csv', help='file to save predictions')
args = parser.parse_args()


NUM_WORDS = 311
NUM_CLASSES = 6

def predict():
    sess = tf.InteractiveSession()
    test_size = count_tfrecord(args.test_file)
    test_dataset = read_tfrecord_dataset(args.test_file)
    test_dataset = test_dataset.map(SampleDecoder(with_label=False)).map(api_one_hot).batch(args.batch_size)
    test_steps = int(np.ceil(test_size / args.batch_size))

    model = tf.keras.models.load_model(args.model_file)

    test_iter = test_dataset.make_one_shot_iterator()
    preds = model.predict(test_iter, verbose=1, steps=test_steps)
    submit_df = pd.DataFrame(preds, columns=['prob' + str(i) for i in range(NUM_CLASSES)])
    submit_df = pd.concat([
        pd.DataFrame({'file_id': range(test_size)}),
        submit_df
    ], axis=1)
    submit_df.to_csv(args.submission_file, index=False)


if __name__ == "__main__":
    predict()
