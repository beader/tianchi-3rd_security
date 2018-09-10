import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPool1D, Conv1D, GlobalMaxPool1D, Concatenate, Dense


def build_cnn_model(num_classes, seq_len, feat_size, name='cnn'):
    input_seq = Input(shape=(seq_len, feat_size), dtype='float32')
    x = MaxPool1D(5, strides=2)(input_seq)

    conv1 = Conv1D(64, kernel_size=3, strides=1, activation='relu')(x)
    conv1 = GlobalMaxPool1D()(conv1)

    conv2 = Conv1D(64, kernel_size=5, strides=2, activation='relu')(x)
    conv2 = GlobalMaxPool1D()(conv2)

    conv3 = Conv1D(64, kernel_size=7, strides=3, activation='relu')(x)
    conv3 = GlobalMaxPool1D()(conv3)

    x = Concatenate()([conv1, conv2, conv3])
    x = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_seq, outputs=x, name=name)
    return model
