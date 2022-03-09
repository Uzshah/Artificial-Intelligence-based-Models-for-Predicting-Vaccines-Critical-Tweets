"""Liberary for training LM and DL Classification models"""
import os
import random
import time
import datetime
import torch
import argparse
import numpy as np
import pandas as pd
from torch.nn import functional as F
from transformers import (get_linear_schedule_with_warmup,AdamW,AutoModel, AutoTokenizer,                   AutoModelForSequenceClassification)
from torch.utils.data import (TensorDataset,DataLoader,RandomSampler, SequentialSampler, Dataset)

class DLClassifier():
    
    def __init__(self, mp, num_classes):
        self.mp = mp
        self.num_classes = num_classes
        self.train_dataloader = None
        self.test_dataloader  = None
        self. model = None
        self. optimizer = None
        self. scheduler = None
        
    def calculate_scores(self, preds, labels):
        pred_flat = np.argmax(np.concatenate(preds), axis=1).flatten()
        results = dict()
        results['precision_score'] = precision_score(labels, pred_flat, average='binary')
        results['recall_score'] = recall_score(labels, pred_flat, average='binary')
        results['f1_score'] = f1_score(labels, pred_flat, average='binary')
        return results


    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))
    
    def convert_label(self, label):
        if label == "c":
            return 1
        elif label == "o":
            return 0 
        elif label == "f":
            return 2
        else:
            raise Exception("label classes must be 'c', 'f' or 'o'")
        
        
    def convert_prediction(self, pred):
        if pred == 1:
            return "c"
        elif pred == 0:
            return "o"
        elif pred == 2:
            return "f"
        else:
            raise Exception("prediction classes must be '0', '1' or 2")
   

    def bert_encode(self, df, tokenizer): 
        input_ids = []
        attention_masks = []
        for sent in df[["text"]].values:
            sent = sent.item()
            encoded_dict = tokenizer.encode_plus(
                                sent,                      
                                add_special_tokens = True, 
                                max_length = 128,           
                                pad_to_max_length = True,
                                truncation = True,
                                return_attention_mask = True,   
                                return_tensors = 'pt',    
                        )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        inputs = {
        'input_word_ids': input_ids,
        'input_mask': attention_masks}

        return inputs
    
    def prepare_dataloaders(self, train_df,test_df,batch_size=8):
        # Load the AutoTokenizer with a normalization mode if the input Tweet is raw

        tokenizer = AutoTokenizer.from_pretrained(self.mp, use_fast=False, normalization=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
#             tokenizer.pad_token_id = tokenizer.eos_token_id
            
        tweet_train = self.bert_encode(train_df, tokenizer)
        tweet_train_labels = train_df.label.astype(int)

        tweet_test = self.bert_encode(test_df, tokenizer)

        input_ids, attention_masks = tweet_train.values()
        labels = torch.tensor(tweet_train_labels.values)
        labels = labels.type(torch.LongTensor)
        train_dataset = TensorDataset(input_ids, attention_masks, labels)


        input_ids, attention_masks = tweet_test.values()
        test_dataset = TensorDataset(input_ids, attention_masks)


        train_dataloader = DataLoader(
                    train_dataset,
                    sampler = RandomSampler(train_dataset), 
                    batch_size = batch_size 
                )


        test_dataloader = DataLoader(
                    test_dataset, 
                    sampler = SequentialSampler(test_dataset), 
                    batch_size = batch_size
                )
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader
        #return train_dataloader, test_dataloader
    
    def prepare_dataloaders_labeling(self, df,batch_size=8):
        # Load the AutoTokenizer with a normalization mode if the input Tweet is raw

        tokenizer = AutoTokenizer.from_pretrained(self.mp, use_fast=False, normalization=True)

#         tweet_train = self.bert_encode(train_df, tokenizer)
#         tweet_train_labels = train_df.label.astype(int)

        tweet_test = self.bert_encode(df, tokenizer)

#         input_ids, attention_masks = tweet_train.values()
#         labels = torch.tensor(tweet_train_labels.values)
#         labels = labels.type(torch.LongTensor)
#         train_dataset = TensorDataset(input_ids, attention_masks, labels)


        input_ids, attention_masks = tweet_test.values()
        test_dataset = TensorDataset(input_ids, attention_masks)




        test_dataloader = DataLoader(
                    test_dataset, 
                    sampler = SequentialSampler(test_dataset), 
                    batch_size = batch_size
                )
        return test_dataloader
    
    def prepare_model(self, model_to_load=None, total_steps=-1):
        
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = self.mp,
            num_labels = self.num_classes,  
            output_attentions = False, 
            output_hidden_states = False,
        )

        optimizer = AdamW(model.parameters(),
                        lr = 5e-5,
                        eps = 1e-8
                        )
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, 
                                                    num_training_steps = total_steps)

        if model_to_load is not None:
            try:
                model.roberta.load_state_dict(torch.load(model_to_load))
                print("LOADED MODEL")
            except:
                pass
        self. model = model
        self. optimizer = optimizer
        self. scheduler = scheduler
#         return model, optimizer, scheduler
    
    
    
    def train(self, epochs):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(0, epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                self.model.zero_grad()        
                outputs = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
            avg_train_loss = total_train_loss / len(self.train_dataloader)            
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        
    def predict(self, test_dataloader):
        self.model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        preds = []

        for batch in test_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            with torch.no_grad():        
                outputs = self.model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask)
                logits = outputs.logits

            logits = logits.detach().cpu().numpy()
            for logit in logits:
                preds.append(logit)

        return preds