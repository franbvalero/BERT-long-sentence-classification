import itertools
import json
import math
import numpy as np
import os
import pandas as pd
import random
import shutil
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, f1_score
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer,BertModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from utils import save_metric_plot, save_confusion_matrix_plot

MODEL_CLASSES = {
    'bert': ("bert-base-uncased", BertForSequenceClassification, BertTokenizer),
    'distilbert': ("distilbert-base-uncased", DistilBertForSequenceClassification, DistilBertTokenizer),
}

def petrained_language_model_classifier(model_name, num_labels, output_attentions=False, output_hidden_states=False, lowercase=True, cache_dir=None):
    petrained_model_name, sequence_classifier_class, tokenizer_class = MODEL_CLASSES[model_name]
    # petrained tokenizer
    model_tokenizer = tokenizer_class.from_pretrained(
        petrained_model_name, 
        do_lower_case=lowercase,
        cache_dir=cache_dir,
    )
    # petrained sequence classifier
    model_classifier = sequence_classifier_class.from_pretrained(
        petrained_model_name,
        num_labels=num_labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_dir=cache_dir,
    )
    return model_tokenizer, model_classifier


class RoBert(nn.Module):

    """
        Recurrences over Bert:
        https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification
    """
    
    def __init__(self, num_labels, bert_pretrained="bert-base-uncased", device="cpu", lstm_hidden_dim=1024, dense_hidden_dim=32, 
        num_lstm_layers=1, num_dense_layers=2):
        super(RoBert, self).__init__()
        self._petrained_language_model = BertModel.from_pretrained(bert_pretrained)
        self._tokenizer = BertTokenizer.from_pretrained(bert_pretrained)
        self.device = device
        self.num_labels = num_labels
        self.size_petrained = self._petrained_language_model.config.hidden_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(self.size_petrained, lstm_hidden_dim, num_lstm_layers, bias=True, batch_first=True)
        self.dropout_plm = nn.Dropout(self._petrained_language_model.config.hidden_dropout_prob)
        self.dropout_lstm = nn.Dropout(self._petrained_language_model.config.hidden_dropout_prob)
        fc_layers = []
        prev_dim = lstm_hidden_dim
        for _ in range(num_dense_layers):
            # init dense layers
            linear_ = nn.Linear(prev_dim, dense_hidden_dim)
            linear_.weight.data.normal_(mean=0.0, std=self._petrained_language_model.config.initializer_range)
            linear_.bias.data.zero_()
            fc_layers.append(linear_)
            prev_dim = dense_hidden_dim
        fc_layers.append(nn.Linear(prev_dim, self.num_labels))
        self.fc = nn.Sequential(*fc_layers)
    
    def _generate_init_hidden_state(self, batch_size):
        h0 = Variable(torch.zeros((self.num_lstm_layers, batch_size, self.lstm_hidden_dim), requires_grad=False).to(self.device))
        c0 = Variable(torch.zeros((self.num_lstm_layers, batch_size, self.lstm_hidden_dim), requires_grad=False).to(self.device))
        return (h0, c0)

    def forward(self, x):
        
        # possible keys: input_ids, attention_mask, token_type_ids, num_segments, labels
        input_id, attention_mask, token_type_ids, num_segments = x['input_ids'], x['attention_mask'], x['token_type_ids'], x['num_segments']
        # [num_sentences, num_segments, segment_size] => [total_segments, segment_size]
        num_sentences, max_segments, segment_length = input_id.size()

        total_segments = num_sentences * max_segments
        input_id_ = input_id.view(total_segments, segment_length)
        attention_mask_ = attention_mask.view(total_segments, segment_length)
        token_type_ids_ = token_type_ids.view(total_segments, segment_length)
        pooler_output = self._petrained_language_model(
                input_ids=input_id_,
                attention_mask=attention_mask_,
                token_type_ids=token_type_ids_,
        )[1] 
        
        pooler_output = self.dropout_plm(pooler_output)
        document_embedding = pooler_output.view(num_sentences, max_segments, self.size_petrained) # [total_segments, size_petrained_model] -> [num_sentences, max_segments, size_petrained]
        
        lstm_outputs, _ = self.lstm(document_embedding, self._generate_init_hidden_state(num_sentences))
        lstm_outputs = self.dropout_lstm(lstm_outputs)        

        idxs_sentences = torch.arange(num_sentences)
        idx_last_output = num_segments - 1
        ouput_last_time_step = lstm_outputs[idxs_sentences, idx_last_output]

        logits = self.fc(ouput_last_time_step)

        if "labels" in x:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), x['labels'].view(-1))
            return loss, logits
        return logits
    
    def get_parameters_to_optimize(self, weight_decay=0.01):
        
        params_plmc = get_optimizer_grouped_parameters(
            list(self._petrained_language_model.named_parameters()), 
            weight_decay=weight_decay
        )
        
        params_fc = get_optimizer_grouped_parameters(
            list(self.fc.named_parameters()), 
            weight_decay=weight_decay
        )
        
        return params_plmc + params_fc

    def tokenizer(self):
        return self._tokenizer
    
    def petrained_language_model(self):
        return self._petrained_language_model

def get_optimizer_grouped_parameters(param_optimizer, weight_decay=0.01, no_decay=["bias", "LayerNorm.weight"]):
    if len(no_decay) > 0:
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
    else:
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            }
        ]
    return optimizer_grouped_parameters

def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def train(model, train_inputs, validation_inputs, device, path_model, id2label, batch_size=128, weight_decay=0.01, num_train_epochs=10, lr=5e-5, eps=1e-8, 
    num_warmup_steps=0, seed=42, decimals=3, max_grad_norm=1, use_token_type_ids=False, lr_lstm=0.001, epochs_decrease_lr_lstm=3, reduced_factor_lstm=0.95):
    
    # Step 0: prapare data loaders and model
    
    train_sampler = RandomSampler(train_inputs)
    train_dataloader = DataLoader(train_inputs, sampler=train_sampler, batch_size=batch_size)
    validation_sampler = SequentialSampler(validation_inputs)
    validation_dataloader = DataLoader(validation_inputs, sampler=validation_sampler, batch_size=batch_size)
    robert = False
    
    model.to(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Step 1: preparere optimizer

    """
        Pre-trained model layers:
        
        - Sometimes practicioners will opt to 'freeze' certain layers when fine-tuning, or to apply different learning rates, 
            apply diminishing learning rates ...
        
        - If your task and fine-tuning dataset is very different from the dataset used to train the transfer learning model,
            freezing the weights may not be a good idea. 
    """

    num_training_batches = len(train_dataloader)
    total_training_steps = num_training_batches * num_train_epochs
    best_val_loss = 100.0
    epoch_wo_improve = 0
    
    if isinstance(model, RoBert):
        optimizer_grouped_parameters = model.get_parameters_to_optimize(weight_decay=weight_decay)
        optimizer_lstm = torch.optim.Adam(model.lstm.parameters(), lr=lr_lstm)
        robert = True
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)
    
    # Step 2: Training loop
    
    training_loss_steps = np.empty((num_train_epochs, num_training_batches))
    training_acc_steps = np.empty_like(training_loss_steps)
    training_f1_steps = np.empty_like(training_loss_steps)
    training_cel_steps = np.empty_like(training_loss_steps)

    validation_loss_steps = np.empty((num_train_epochs, len(validation_dataloader)))
    validation_cel_steps = np.empty_like(validation_loss_steps)
    validation_f1_steps = np.empty_like(training_loss_steps)
    validation_acc_steps = np.empty_like(validation_loss_steps)

    best_validation_acc = 0.0
    best_model = None
    num_class_labels = len(id2label)
    labels = [i for i in range(num_class_labels)]
    class_labels = list(id2label.values())
    best_confussion_matrix = None

    set_seed(random_seed=seed)
    
    for idx_epoch in range(0, num_train_epochs):
        
        """ Training """
        
        model.train()
        stime_train_epoch = time.time()

        for idx_train_batch, train_batch in enumerate(train_dataloader):
            # 0: input_ids, 1: attention_mask, 2:token_type_ids, 3:num_segments, 4: labels
            batch_train = tuple(data_.to(device) for data_ in train_batch)
            inputs = {
                'input_ids': batch_train[0],
                'attention_mask': batch_train[1],
                'labels': batch_train[-1],
            }
            if use_token_type_ids:
                inputs['token_type_ids'] = batch_train[2]

            optimizer.zero_grad()
            
            if robert:
                inputs['num_segments'] = batch_train[3]
                loss, logits = model(inputs)
            else:
                loss, logits = model(**inputs)
            
            if n_gpu > 1:
                loss = loss.mean()
                
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm) # clipping gradient for avoiding exploding gradients
            optimizer.step()
            scheduler.step()
            if robert:
                optimizer_lstm.step()
            
            logits = logits.detach().cpu().numpy()
            hypothesis = np.argmax(logits, axis=1)
            expected_predictions = inputs['labels'].to('cpu').numpy()

            training_loss_steps[idx_epoch, idx_train_batch] = loss.item()
            training_acc_steps[idx_epoch, idx_train_batch] = accuracy_score(expected_predictions, hypothesis)
            training_f1_steps[idx_epoch, idx_train_batch] = f1_score(expected_predictions, hypothesis, average='macro')
            training_cel_steps[idx_epoch, idx_train_batch] = log_loss(expected_predictions, logits, labels=labels)
        
        ftime_train_epoch = time.time()
        print(f"Training - Epoch {idx_epoch},\n\tLoss: {np.mean(training_loss_steps[idx_epoch,:]):.3f}\n\tCEL: {np.mean(training_cel_steps[idx_epoch,:]):.3f}\n\tAccuracy:{np.mean(training_acc_steps[idx_epoch, :]):.3f}\n\tF1:{np.mean(training_f1_steps[idx_epoch, :]):.3f}\n\ttime:{(ftime_train_epoch - stime_train_epoch)/60:.2f}")
        
        """ Validation"""
        
        model.eval()
        current_confussion_matrix = np.zeros((num_class_labels, num_class_labels), dtype=int)
        stime_validation_epoch = time.time()
        
        for idx_validation_batch, validation_batch in enumerate(validation_dataloader):
            
            batch_validation = tuple(data_.to(device) for data_ in validation_batch)
            inputs = {
                'input_ids': batch_validation[0],
                'attention_mask': batch_validation[1],
                'labels' : batch_validation[-1],
            }
            if use_token_type_ids:
                inputs['token_type_ids'] = batch_validation[2]
            
            with torch.no_grad():

                if robert:
                    inputs['num_segments'] = batch_validation[3]
                    loss, logits = model(inputs)
                else:
                    loss, logits = model(**inputs)
            
            if n_gpu > 1:
                loss = loss.mean()
            
            logits = logits.detach().cpu().numpy()
            hypothesis = np.argmax(logits, axis=1)
            expected_predictions = inputs['labels'].to('cpu').numpy()

            current_confussion_matrix += confusion_matrix(expected_predictions, hypothesis, labels=labels)
            validation_loss_steps[idx_epoch, idx_validation_batch] = loss.item()
            validation_cel_steps[idx_epoch, idx_validation_batch] = log_loss(expected_predictions, logits, labels=labels)
            validation_acc_steps[idx_epoch, idx_validation_batch] = accuracy_score(expected_predictions, hypothesis)
            validation_f1_steps[idx_epoch, idx_validation_batch] = f1_score(expected_predictions, hypothesis, average='macro')
        
        current_validation_acc = np.mean(validation_acc_steps[idx_epoch, :])
        current_val_loss = np.mean(validation_loss_steps[idx_epoch, :])
        ftime_validation_epoch = time.time()

        print(f"Validation - Epoch {idx_epoch},\n\tLoss: {current_val_loss:.3f}\n\tCEL: {np.mean(validation_cel_steps[idx_epoch, :]):.3f}\n\tAccuracy:{current_validation_acc:.3f}\n\tF1:{np.mean(validation_f1_steps[idx_epoch, :]):.3f}\n\ttime:{(ftime_validation_epoch - stime_validation_epoch)/60:.2f}")
    
        if current_validation_acc > best_validation_acc:
            best_validation_acc = current_validation_acc
            best_model = model
            best_confussion_matrix = current_confussion_matrix
        if robert:
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
            else:
                epoch_wo_improve += 1
                if epoch_wo_improve > epochs_decrease_lr_lstm:
                    for g in optimizer_lstm.param_groups:
                        g['lr'] = g['lr'] * reduced_factor_lstm
                    epoch_wo_improve = 0
                    print("Updated lr lstm")

    print(f"Best validation accuracy: {best_validation_acc:.3f}")
    
    if os.path.exists(path_model):
        shutil.rmtree(path_model)  
    os.makedirs(path_model)
    
    """ Save model"""
    if robert:
         torch.save(best_model.state_dict(), os.path.join(path_model, "pytorch_model.bin"))
    else:
        model_to_save = best_model.module if hasattr(best_model, 'module') else best_model
        model_to_save.save_pretrained(path_model)
    
    """ Save metric values"""
    
    metrics_values = np.round(
        np.array([
            np.mean(training_loss_steps, axis=1),
            np.mean(training_cel_steps, axis=1),
            np.mean(training_acc_steps, axis=1),
            np.mean(training_f1_steps, axis=1),
            np.mean(validation_loss_steps, axis=1),
            np.mean(validation_cel_steps, axis=1),
            np.mean(validation_acc_steps, axis=1),
            np.mean(validation_f1_steps, axis=1),
        ]), 
        decimals=decimals,
    )
    
    metrics_labels = [
        "training_loss",
        "training_cel",
        "training_acc",
        "training_f1",
        "validation_loss",
        "validation_cel",
        "validation_acc",
        "validation_f1",
    ]
    
    df = pd.DataFrame(
        metrics_values.T,
        columns=metrics_labels,
    )
    
    df.to_csv(os.path.join(path_model, f"metrics.csv"), sep=',', encoding='utf-8', index=False, header=True)
    
    """ Save information plots"""
    
    save_metric_plot(os.path.join(path_model, "accuracy"), df["training_acc"], df["validation_acc"], "Epoch", "Accuracy", loc="lower right")
    save_metric_plot(os.path.join(path_model, "loss"), df["training_loss"], df["validation_loss"], "Epoch", "Loss", loc="upper right")
    save_metric_plot(os.path.join(path_model, "cel"), df["training_cel"], df["validation_cel"], "Epoch", "CEL", loc="upper right")
    save_metric_plot(os.path.join(path_model, "f1"), df["training_f1"], df["validation_f1"], "Epoch", "F1", loc="lower right")
    
    np.savetxt(os.path.join(path_model, "confusion.txt"), best_confussion_matrix, fmt="%i")
    save_confusion_matrix_plot(os.path.join(path_model, "confusion"), best_confussion_matrix, class_labels)
    
    id_to_label_str = json.dumps(id2label)
    with open(os.path.join(path_model, "labels.json"), "w") as fjson:
        fjson.write(id_to_label_str)