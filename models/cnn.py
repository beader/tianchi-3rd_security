from keras.layers import Input, Conv1D, MaxPool1D, GlobalMaxPool1D, Activation, Concatenate, Dense
from keras.models import Model


def build_cnn_model(num_classes, seq_len, feat_size):
    input_seq = Input(shape=(seq_len, feat_size), dtype='float32')
    x = MaxPool1D(pool_size=5, strides=2)(input_seq)

    conv1 = Conv1D(filters=64, kernel_size=3, strides=1)(x)
    conv1 = Activation('relu')(conv1)
    conv1 = GlobalMaxPool1D()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=5, strides=2)(x)
    conv2 = Activation('relu')(conv2)
    conv2 = GlobalMaxPool1D()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=7, strides=3)(x)
    conv3 = Activation('relu')(conv3)
    conv3 = GlobalMaxPool1D()(conv3)

    x = Concatenate()([conv1, conv2, conv3])
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_seq, outputs=x)
    return model
