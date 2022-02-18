import argparse
import os
import json
import random

import numpy as np
import tensorflow as tf

"""
TFRecords use serialization and deserialization to manage 
files in a more efficient way

TFRecords are composed of Examples, and each Example is composed
of Features
"""


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def _float32_image_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))




def create_example(sketch, label):
    """
    Creates a tf.train.Example out of the sketches from the
    QuickDraw dataset

    image : strokes (delta x, delta y, binary pen state)
    label : class of the given sketch
    """

    features = {
        'sketch' : _float32_image_feature(sketch),
        'sketch_size' : _int64_feature(sketch.shape[0]),
        'label' : _int64_feature(label)
    }

    # converting features into a single example
    return tf.train.Example(features=tf.train.Features(feature=features))

def convert_dataset_in_chunks(set_type, idx_to_classes, class_files, n_chunks, datapath):
    """
    Function used to convert to dataset into the TFRecord format in
    chunks

    set_type : type of data (training, validation, test)
    idx_to_classes : index of each class in the class list
    class_files : path for each class file
    n_chunks : number of chunks the data will be divided into
    datapath : path of where the converted files will be stored
    """

    # dividing the TFRecords in 1 or more chunks
    for chunk in range(0, n_chunks):
        # opening the data files
        tf_record_shard_path = os.path.join(datapath, "{}{:03}.records".format(set_type, chunk))
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(tf_record_shard_path, options) as writer:

            # this loop is used to collect the data from each class
            # and then shuffling them up so as to prevent the dataset
            # from being entirely sequential
            class_shards = []
            for i, class_name in enumerate(idx_to_classes):
                # loading the data from a certain class
                data = np.load(class_files[class_name],
                               encoding='latin1', allow_pickle=True, mmap_mode='r')
                print(f"Loading chunk {chunk + 1}/{n_chunks} from class {class_name} (class {i + 1}/{len(idx_to_classes)})...")

                # getting important info
                n_samples = len(data[set_type]) // n_chunks
                start = n_samples * chunk
                end = start + n_samples
                samples = data[set_type][start:end]
                
                labels = np.ones((n_samples,), dtype=int) * i

                class_shards.append((samples, labels))
            

            # shuffle the shards
            random.shuffle(class_shards)

            # take one sample from each shard at a time, mixing all of them
            # 345 classes with 2500 sketches each
            for samples, labels in class_shards:
                cur_index = 0
                # print(samples.shape)
                for sample in samples:
                    tf_example = create_example(sample, labels[cur_index])
                    serialized = tf_example.SerializeToString()
                    writer.write(serialized)

                    # print(tf.train.Example.FromString(serialized))
                    # s, l = parse_quickdraw_image(serialized)
                    # print('Resultados:',s, l)
                    print("Saved {}/{} samples from each class shard".format(cur_index+1, n_samples))
                    cur_index += 1
            


def parse_quickdraw_image(example): 
    """
    Function used to parse the TFRecords Examples 
    via deserialization, returning the reconstructed
    image and its corresponding label

    example : TFRecord example
    """

    image_feature_description = {
        'sketch' : tf.io.VarLenFeature(tf.float32),
        'sketch_size' : tf.io.FixedLenFeature([], tf.int64),
        'label' : tf.io.FixedLenFeature([], tf.int64)
    }

    # parsing and reconstructing the sketch
    parsed_example = tf.io.parse_single_example(example, image_feature_description)
    sketch = tf.reshape(parsed_example['sketch'].values, [parsed_example['sketch_size'], 3])

    return sketch, parsed_example['label']

def parse_dataset(filename):
    raw_dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP')

    parsed_dataset = raw_dataset.map(parse_quickdraw_image)

    return parsed_dataset

def main():
    # argument parser responsible for collecting key info for the data extraction and conversion
    parser = argparse.ArgumentParser(description='dataset convertion to TFRecord format')
    parser.add_argument('--dataset-dir', type=str, default='/store/shared/datasets/quickdraw')
    parser.add_argument('--class-list', type=str, default='/store/lshimabucoro/projects/bumblebee/prep_data/quickdraw/list_quickdraw.txt')
    parser.add_argument('--n-chunks', type=int, default=50)
    parser.add_argument('--target-dir', type=str, default='/store/lshimabucoro/projects/bumblebee/scratch/datasets/quickdraw_raw_345/')
    args = parser.parse_args()

    # if the given path does not exist then we create a new directory with that name
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)
    
    # creating class-file dicts in order to manipulate the .npz files
    # later to convert them
    class_files, idx_to_classes = {}, []
    with open(args.class_list, 'r') as class_list_file:
        for class_name in class_list_file:
            class_name = class_name.rstrip()
            class_files[class_name] = f"{args.dataset_dir}/{class_name}.npz".format(args.dataset_dir, class_name)
            idx_to_classes.append(class_name)
    
    classes_to_idx = {c: i for i, c in enumerate(idx_to_classes)}

    # class_files = class:file_path (dict)
    # idx_to_classes = list of classes
    # classes_to_idx = class:index (dict)

    # stores metadate in json file
    metadata = {"idx_to_classes": classes_to_idx, "classes_to_idx": classes_to_idx}
    with open(os.path.join(args.target_dir, "meta.json"), 'w') as outfile:
        json.dump(metadata, outfile)
    
    # calling the conversion functions to store the data in the TFRecord format
    # convert_dataset_in_chunks('valid', idx_to_classes, class_files, 1, args.target_dir)
    # convert_dataset_in_chunks('test', idx_to_classes, class_files, 1, args.target_dir)
    # convert_dataset_in_chunks('train', idx_to_classes, class_files, 10, args.target_dir)
    dataset = parse_dataset('/store/lshimabucoro/projects/bumblebee/scratch/datasets/quickdraw_raw_345/valid000.records').shuffle(buffer_size=345*2500)
    for record in dataset.take(10):
        print(record, '\n')

if __name__ == '__main__':
    main()