#!/usr/bin/env python
# coding: utf-8

# ###########################
# 
# This notebook is intended to be used for load and decode local Tfrecord datasets
# 
# Please make sure image data are loaded locally before running the code.
# 
# **Credits to following link for this notebook.**
# 
# Functions are based on the referenced notebook!
# 
# * [Kaggle metal to petal challenge](https://www.kaggle.com/c/tpu-getting-started)
# * [Robert Border](https://www.kaggle.com/rborder/tpu-flower-classification?kernelSessionId=78320658)
# * [Khoa Phung](https://www.kaggle.com/khoaphng/model-efficientnetb7?kernelSessionId=76625228)
# * [Tensorflow load and proprocessing images](https://www.tensorflow.org/tutorials/load_data/images)
# 
# #############################

# ## Imports

# In[14]:


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.utils import image_dataset_from_directory
import re
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


# In[3]:


print(tf.__version__)
print(tf.keras.__version__)


# In[16]:


#Detect TPU, return appropriate distribution strategy
def tpu_detect():
   try:
       tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
       print('Running on TPU ', tpu.master())
   except ValueError:
       tpu = None

   if tpu:
       tf.config.experimental_connect_to_cluster(tpu)
       tf.tpu.experimental.initialize_tpu_system(tpu)
       strategy = tf.distribute.experimental.TPUStrategy(tpu)
   else:
       strategy = tf.distribute.get_strategy() 

   print("REPLICAS: ", strategy.num_replicas_in_sync)
   return strategy


# In[18]:


#############
"""
CONSTANT
"""
############
# The class names for flowers with the corresponding index number
CLASS= ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']

# Data file paths
PATH='./image_data_224x224'
TRAIN_FILES=tf.io.gfile.glob(PATH+'/train/*.tfrec')
TEST_FILES=tf.io.gfile.glob(PATH+'/test/*.tfrec')
VAL_FILES=tf.io.gfile.glob(PATH+'/val/*.tfrec')

#Image size
IMAGE_SIZE=[224,224]
BATCH_SIZE = 16 *tpu_detect().num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE


# In[6]:


#define a function to load Tfrecord data from local directories
def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


# In[7]:


#decode jpeg image and normlize the image and reshape the image 
def decode_normalize(image_file):
    img=tf.image.decode_jpeg(image_file, channels=3)
    img=tf.cast(img, tf.float32)/255.0
    img=tf.reshape(img, [*IMAGE_SIZE, 3])
    return img


# In[8]:


# read train and val with label
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_normalize(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs


# In[9]:


#read test data without label
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # for test data with no label but only id names, and the class is our target prediction
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_normalize(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)


# In[26]:


def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO)
    # statement in the next function (below), this happens essentially
    # for free on TPU. Data pipeline code is executed on the "CPU"
    # part of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0, 2)
    return image, label   

def get_training_dataset(dataset):
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.cache() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(dataset):
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(test_files, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


# In[21]:


def data_summary(train=TRAIN_FILES, validation=VAL_FILES, tes=TEST_FILES):
    train_img_total = count_data_items(TRAIN_FILES)
    val_img_total = count_data_items(VAL_FILES)
    test_img_total = count_data_items(TEST_FILES)
    print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(train_img_total, val_img_total, test_img_total))


# In[22]:


def get_steps_per_epochs():
    STEPS_PER_EPOCH=count_data_items(TRAIN_FILES)//BATCH_SIZE
    print(STEPS_PER_EPOCH)
    return STEPS_PER_EPOCH


# In[25]:


def check_image_label_shape(data):    
    np.set_printoptions(threshold=15, linewidth=80)

    print(f"The {data} data shapes:")
    if data in [train, val]:
        for image, label in data.take(0):
            print(image.numpy().shape, label.numpy().shape)
        print(f"{data} data label examples:", label.numpy())
    elif data == test:
        print("Test data shapes:")
        for image, idnum in data.take(0):
            print(image.numpy().shape, idnum.numpy().shape)
        print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string   
        


# In[ ]:




