import tensorflow as tf


def read_tfrecord_dataset(path):
    if path.endswith('.gz'):
        compression_type = 'GZIP'
    else:
        compression_type = None
    return tf.data.TFRecordDataset(path, compression_type=compression_type)


class SampleDecoder(object):
    def __init__(self, with_label=True):
        self.with_label = with_label
        
    def __call__(self, serialized_example):
        features = {
            'file_id': tf.FixedLenFeature([], tf.int64),
            'seq_len': tf.FixedLenFeature([], tf.int64),
            'api_cnt': tf.FixedLenFeature([], tf.int64),
            'api': tf.VarLenFeature(tf.int64),
            'index': tf.VarLenFeature(tf.int64)
        }
        if self.with_label:
            features['label'] = tf.FixedLenFeature([], tf.int64)
        example = tf.parse_single_example(serialized_example, features=features)
        return example
    

def reweighted_filter(reweighted_ratios):
    def wrapper(example):
        return tf.random_uniform([], 0, 1) < reweighted_ratios[example['label']]
    return wrapper


def api_one_hot(example):
    indices = tf.stack([example['index'].values, example['api'].values], axis=1)
    sp_vals = tf.ones(example['api_cnt'])
    ohe = tf.sparse_to_dense(indices, [5001, 311], sp_vals)
    if 'label' in example:
        return ohe, [example['label']]
    else:
        return ohe, [0]


def count_tfrecord(path):
    if path.endswith('.gz'):
        compress_type = tf.python_io.TFRecordCompressionType.GZIP
    else:
        compress_type = tf.python_io.TFRecordCompressionType.NONE
    options = tf.python_io.TFRecordOptions(compress_type)
    record_iter = tf.python_io.tf_record_iterator(path, options)
    i = 0
    for _ in record_iter:
        i += 1
    return i


def make_reweighted_dataset(dataset, batch_size, reweight_ratios_input):
    dataset = (dataset
              .map(SampleDecoder(with_label=True))
              .filter(reweighted_filter(reweight_ratios_input))
              .map(api_one_hot)
              .shuffle(buffer_size=500)
              .batch(batch_size))
    return dataset
