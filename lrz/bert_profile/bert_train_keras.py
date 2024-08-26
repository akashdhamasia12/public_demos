import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import numpy as np
import time
import os

# os.environ[" TF_ENABLE_ONEDNN_OPTS"] = "0"
# print(os.environ[" TF_ENABLE_ONEDNN_OPTS"])

df = pd.read_csv("spam.csv")

df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
# df.head()

X_train, X_test, y_train, y_test = train_test_split(df['Message'],df['spam'], stratify=df['spam'])

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='softmax', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])

model.summary()

#Callback class for time history (picked up this solution directly from StackOverflow)

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
time_callback = TimeHistory()

# Function for approximate training time with each layer independently trained

def get_average_layer_train_time(epochs):
    
    #Loop through each layer setting it Trainable and others as non trainable
    results = []
    for i in range(len(model.layers)):
        
        layer_name = model.layers[i].name    #storing name of layer for printing layer
        
        #Setting all layers as non-Trainable
        for layer in model.layers:
            layer.trainable = False
            
        #Setting ith layers as trainable
        model.layers[i].trainable = True
        
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        #Fit on a small number of epochs with callback that records time for each epoch
        model.fit(X_train[:100], y_train[:100], epochs=epochs, verbose=1, callbacks = [time_callback])

        results.append(np.average(time_callback.times))
        #Print average of the time for each layer
        print(f"{layer_name}: Approx (avg) train time for {epochs} epochs = ", np.average(time_callback.times))
    return results

runtimes = get_average_layer_train_time(2)
# plt.plot(runtimes)