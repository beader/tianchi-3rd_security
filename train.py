import os
import time
import pandas as pd
import argparse
from data_utils import APISequence, train_test_split
from models import build_cnn_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

parser = argparse.ArgumentParser(description="Train classifier")
parser.add_argument('--train-pkl-file', default='./data/train_encoded_api_seqs.pkl',
                    help='pickled train data')
parser.add_argument('--max-seq-len', type=int, default=5001,
                    help='max sequence length')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=30,
                    help='max epochs to run')
parser.add_argument('--log-dir', default='./logs')
parser.add_argument('--validate-split', type=float, default=0.1,
                    help='ratio of validate data')
parser.add_argument('--patience', type=float, default=5,
                    help='early stopiing patience')
args = parser.parse_args()


NUM_WORDS = 311
NUM_CLASSES = 6

if __name__ == "__main__":
    print('reading train pkl file')
    df = pd.read_pickle(args.train_pkl_file)
    api_seq = APISequence(df['api_seq'].values, args.max_seq_len, NUM_WORDS)
    labels = df['label'].astype('int').values
    tr_seq, val_seq = train_test_split(api_seq, labels, batch_size=args.batch_size,
                                       validation_split=args.validate_split,
                                       reweight_ratios={0: 0.027})
    model = build_cnn_model(NUM_CLASSES, args.max_seq_len, NUM_WORDS)
    model.name = 'cnn_{}'.format(int(time.time()))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    log_dir = os.path.join('./logs', model.name)
    model_filename = model.name + '.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_save_path = os.path.join(log_dir, model_filename)
    callbacks = [
        TensorBoard(log_dir=log_dir, batch_size=args.batch_size),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience)
    ]
    print('start training')
    model.fit_generator(tr_seq, epochs=args.epochs, validation_data=val_seq, callbacks=callbacks,
                        workers=2, use_multiprocessing=True)
