import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import numpy as np
import os
import time

# os.environ[" TF_ENABLE_ONEDNN_OPTS"] = "0"
# print(os.environ[" TF_ENABLE_ONEDNN_OPTS"])

df = pd.read_csv("spam.csv")

df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
# df.head()

X_train, X_test, y_train, y_test = train_test_split(df['Message'],df['spam'], stratify=df['spam'])

bert_preprocess = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
bert_encoder = transformers.BertModel.from_pretrained("bert-base-uncased")


class BertClassifier(nn.Module):

    def __init__(self, bert_encoder, bert_preprocess):

        super(BertClassifier, self).__init__()

        self.bert_encoder = bert_encoder
        self.bert_preprocess = bert_preprocess
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 1)

    # def forward(self, input_id, mask):

    #     _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
    #     dropout_output = self.dropout(pooled_output)
    #     linear_output = self.linear(dropout_output)
    #     final_layer = self.relu(linear_output)

    #     return final_layer

    def forward(self, text):
        input_ids = self.bert_preprocess.encode(text, return_tensors="pt").input_ids
        attention_mask = self.bert_preprocess.encode(text, return_tensors="pt").attention_mask
        outputs = self.bert_encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        l = self.dropout(pooled_output)
        l = self.fc(l)
        l = torch.sigmoid(l)
        # l = torch.nn.Softmax(l)

        return l

model = BertClassifier(bert_encoder, bert_preprocess)




# class SpamClassifier(nn.Module):
#     def __init__(self, bert_preprocess, bert_encoder):
#         super(SpamClassifier, self).__init__()
#         self.bert_preprocess = bert_preprocess
#         self.bert_encoder = bert_encoder
#         self.dropout = nn.Dropout(0.1)
#         self.output = nn.Linear(768, 1)
        
#     def forward(self, text_input):
#         preprocessed_text = self.bert_preprocess(text_input)
#         outputs = self.bert_encoder(preprocessed_text)
#         x = self.dropout(outputs['pooled_output'])
#         x = self.output(x)
#         return x

# model = SpamClassifier(bert_preprocess, bert_encoder)
        
def get_average_layer_train_time(model, X_train, y_train, epochs):
    results = []

    print(len(model._modules))


    for i in range(len(model._modules)):
        # layer_name = model._modules[i].
        layer_name = "bla"
        for param in model.parameters():
            param.requires_grad = False
        # for param in model._modules[i].parameters():
        #     param.requires_grad = True
            
        criterion = torch.nn.BCELoss()
        # optimizer = torch.optim.Adam(model.layers[i].parameters())
        optimizer = torch.optim.Adam(model.parameters())
        
        timer_list = []
        for epoch in range(epochs):
            start_time = time.time()
            outputs = model(X_train[:100])
            loss = criterion(outputs, y_train[:100])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            duration = time.time() - start_time
            timer_list.append(duration)         

        results.append(np.average(timer_list))
        print(f"{layer_name}: Approx (avg) train time for {epochs} epochs = {np.average(timer_callback.times)}")
    return results

runtimes = get_average_layer_train_time(model, X_train, y_train, 2)
