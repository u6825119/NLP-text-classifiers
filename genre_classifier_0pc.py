
# NOTE: This file contains is a very poor model which looks for manually 
# chosen keywords and if none are found it predicts randomly according
# to the class distribution in the training set

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import constant
from keras.layers import Dense, Dropout, GlobalMaxPool1D, BatchNormalization
from keras import callbacks
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AutoTokenizer
from transformers import TFAutoModel
import tensorflow_addons as tfa

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from
#del train_data['docid']

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']

sequence_len = 300
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert_model = TFAutoModel.from_pretrained("bert-base-cased")


def Classifier_model():
    ##inputs: [ids, mask]
    input_ids = tf.keras.layers.Input(shape = (sequence_len,), name = 'input_ids', dtype = 'int32')
    attention_mask = tf.keras.layers.Input(shape = (sequence_len,), name = 'attention_mask', dtype = 'int32')

    embeddings = bert_model(input_ids, attention_mask = attention_mask)[0] ##[1] = pool output

    net = GlobalMaxPool1D()(embeddings)
    net = BatchNormalization()(net)
    net = Dense(500, activation = 'relu')(net)
    net = Dense(500, activation = 'relu')(net)
    output = Dense(4, activation = 'softmax', name = 'outputs')(net)

    return keras.Model(inputs = [input_ids, attention_mask], outputs = output)    
  
def text_encoding(data):
    ids = np.zeros((len(data), sequence_len))
    mask = np.zeros((len(data), sequence_len))
    
    for i, sequence in enumerate(data):
        tokens = tokenizer.encode_plus(
            text=sequence, 
            add_special_tokens=True,  
            max_length = sequence_len,  
            pad_to_max_length=True,  
            truncation=True,
            return_attention_mask = True,  
            return_tensors = 'tf' ,
            )
        ids[i, :], mask[i, :] =  tokens['input_ids'], tokens['attention_mask']    
    return ids, mask


def training_map_func(input_ids, masks, labels):
    return{'input_ids': input_ids, 'attention_mask': masks}, labels




#pre-processing train data
  
train_ids, train_mask = text_encoding(X)
#one-hot encoding of labels
arr = np.array(Y)
labels = np.zeros((arr.size, 4))
labels[np.arange(arr.size), arr] = 1

train_dataset = tf.data.Dataset.from_tensor_slices((train_ids, train_mask, labels))
train_dataset = train_dataset.map(training_map_func)
train_dataset = train_dataset.shuffle(10000).batch(32)
# train-validation split
dataset_len = len(list(train_dataset))
final_train_ds = train_dataset.take(round(dataset_len*0.9))
final_val_ds = train_dataset.skip(round(dataset_len*0.9))

#pre-processing test data
test_ids, test_mask = text_encoding(Xt)
final_test_ids = tf.convert_to_tensor(constant(test_ids), dtype=tf.float64)
final_test_masks = tf.convert_to_tensor(constant(test_mask), dtype=tf.float64)

# Build the model
model = Classifier_model()
model.layers[2].trainable = False

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.015)
loss = keras.losses.CategoricalCrossentropy()
#metrics = keras.metrics.CategoricalAccuracy()
metrics = tfa.metrics.F1Score(num_classes=4,average="macro",threshold=None)



earlystopping = callbacks.EarlyStopping(monitor ="f1_score", 
                    mode ="max", patience = 10, 
                    restore_best_weights = True) 

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
  
model.fit(final_train_ds, batch_size=64, validation_data = final_val_ds, epochs = 140, callbacks =[earlystopping])


# predict on the test data
pred = model.predict([final_test_ids, final_test_masks])
Y_test_pred = np.argmax(pred, axis = 1)

# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the predction as an integer
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()