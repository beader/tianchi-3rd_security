import os
import time
import argparse
import numpy as np
import tensorflow as tf
from models import build_cnn_model
from data_utils import read_tfrecord_dataset, make_reweighted_dataset, count_tfrecord
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

parser = argparse.ArgumentParser(description="Train classifier")
parser.add_argument('--train-file', required=True, help='TFRecord file for training.')
parser.add_argument('--valid-file', required=True, help='TFRecord file for validation.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Max epochs to run')
parser.add_argument('--log-dir', default='./logs')
parser.add_argument('--patience', type=float, default=5, help='Early stopiing patience')
args = parser.parse_args()


NUM_WORDS = 311
NUM_CLASSES = 6
SEQ_LEN = 5001
LABEL_DIST = [111545, 287, 744, 598, 53, 3397]
REWEIGHT_RATIO = [0.027, 1, 1, 1, 1, 1]


def train():
    sess = tf.InteractiveSession()
    train_size = count_tfrecord(args.train_file)
    valid_size = count_tfrecord(args.valid_file)
    label_distribution = np.array(LABEL_DIST) / np.sum(LABEL_DIST)
    reweight_ratios = np.array(REWEIGHT_RATIO)

    train_dataset = read_tfrecord_dataset(args.train_file)
    valid_dataset = read_tfrecord_dataset(args.valid_file)
    reweight_ratios_input = tf.placeholder(tf.float32, shape=[NUM_CLASSES])

    train_dataset = make_reweighted_dataset(train_dataset,
                                            batch_size=args.batch_size,
                                            reweight_ratios_input=reweight_ratios_input).repeat()
    valid_dataset = make_reweighted_dataset(valid_dataset,
                                            batch_size=args.batch_size,
                                            reweight_ratios_input=reweight_ratios_input).repeat()
    train_steps_per_epoch = int(np.sum(label_distribution * reweight_ratios) * train_size / args.batch_size)
    valid_steps_per_epoch = int(np.sum(label_distribution * reweight_ratios) * valid_size / args.batch_size)

    model_name = 'cnn_{}'.format(int(time.time()))
    model = build_cnn_model(NUM_CLASSES, SEQ_LEN, NUM_WORDS, model_name)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
    log_dir = os.path.join('./logs', model.name)
    model_filename = model.name + '.{epoch:02d}-{val_loss:.4f}.h5'
    model_save_path = os.path.join(log_dir, model_filename)

    callbacks = [
        TensorBoard(log_dir=log_dir, batch_size=args.batch_size),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience)
    ]
    print('start training')
    train_iter = train_dataset.make_initializable_iterator()
    valid_iter = valid_dataset.make_initializable_iterator()
    sess.run([train_iter.initializer, valid_iter.initializer],
             feed_dict={reweight_ratios_input: [0.027, 1, 1, 1, 1, 1]})
    model.fit(train_iter,
              epochs=args.epochs,
              steps_per_epoch=train_steps_per_epoch,
              validation_data=valid_iter,
              validation_steps=valid_steps_per_epoch,
              callbacks=callbacks)


if __name__ == "__main__":
    train()

