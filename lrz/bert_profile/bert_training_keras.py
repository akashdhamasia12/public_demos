import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel

datapath = f'bbc-text.csv'
df = pd.read_csv(datapath)
df.head()

# df.groupby(['category']).size().plot.bar()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }

class Dataset(tf.keras.utils.Sequence):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="tf") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    
class BertClassifier(tf.keras.Model):

    def __init__(self, dropout=0.5, **kwargs):

        super(BertClassifier, self).__init__(**kwargs)

        self.bert = TFBertModel.from_pretrained('bert-base-cased')
        self.dropout = tf.keras.layers.Dropout(dropout)
        # self.linear = tf.keras.layers.Dense(768, 5)
        self.linear = tf.keras.layers.Dense(5, activation='relu')
        self.relu = tf.keras.activations.relu

    def call(self, input_id, attention_mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=attention_mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer



np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

print(len(df_train),len(df_val), len(df_test))

EPOCHS = 5
model = BertClassifier()
LR = 1e-6
              
# train(model, df_train, df_val, LR, EPOCHS)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_dataset = Dataset(df_train)
val_dataset = Dataset(df_val)


train_ = tf.data.Dataset.from_tensor_slices(train_dataset).batch(1)#.shuffle(len(train_dataset))

# tokenized_dataset = df_train.map(
#     Dataset, batched=True, num_proc=1,
# )

# train = train_dataset.to_tf_dataset(
#     columns=["input_ids", "attention_mask"],
#     label_cols=["labels"],
#     batch_size=2,
#     shuffle=True,)

# history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
history = model.fit(train_, epochs=EPOCHS)
# model.fit(X_train[:100], y_train[:100], epochs=epochs, verbose=1, callbacks = [time_callback])
