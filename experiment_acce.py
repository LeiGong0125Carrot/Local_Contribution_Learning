
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertConfig, BertModel
import os
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import torch.nn.functional as F
from torch import where, rand_like, zeros_like, log, sigmoid
from torch.nn import Module
# from settings import *
# # import torch.cuda.amp.GradScaler
# from settings import *
import numpy as np
from transformers import AutoTokenizer, BertModel
from transformers import  DistilBertModel, DistilBertTokenizerFast
import logging
from scipy.spatial.distance import cdist
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold
from utils import *
# from model_ours import *

from PLens import *
import gc
# from gpu_mem_track import MemTracker
from torch.cuda.amp import autocast
import os 

import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from prototype_generate_utility import generate_prototype
from torch.cuda.amp import GradScaler, autocast
# from Gumbel_Sigmoid import *
from accelerate import Accelerator
from torch.distributed import broadcast


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def gold_eval(model, data_loader, p_sent2pid, device, epo_num):
    sentence_pool = pd.read_csv("/home/bwei2/ProtoTextClassification/Data/IMDB_cluster_" + str(self.num_prototypes) + "_to_sub_sentence.csv", index_col=0, header=None).to_numpy()
    model.eval()
    predictions = []
    actual_labels = []
    document = docx.Document()
    # with torch.no_grad():
    for batch in data_loader:
        original_text = batch['original_text']
        input_ids = batch['input_ids'].to(device)
        input_attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        label = batch['label_text']
        pid = [p_sent2pid[i] for i in label]
        attention_mask_label = batch["attention_mask_label"].to(device)
        special_tokens_mask_label = batch["special_tokens_mask_label"].to(device)
        input_ids_label = batch["input_ids_label"].to(device)
        # logging.info(special_tokens_mask_label)
        label_ids = (1 - special_tokens_mask_label) * input_ids_label
        # Initialize the label mask with zeros
        label_mask = torch.zeros_like(input_ids)
        # non_zero_elements = label_ids[label_ids != 0]
        seq = []
        for i in range(label_ids.shape[0]):
            # Get non-zero elements from label_ids for this sequence
            non_zero_elements = label_ids[i][label_ids[i] != 0]
            seq.append(non_zero_elements)
            # Check if the non-zero elements exist in input_ids
            for elem in non_zero_elements:
                # Get the index where the element appears in input_ids
                indices = torch.where(input_ids[i] == elem)[0]
                # Set these positions in the label mask to 1
                label_mask[i][indices] = 1
        # label_mask = find_and_replace_subsequence(label_mask, seq)

        # words, _, df, vocab = model.get_words(original_text)
        # label_words_in_order = []
        # for label_text in label:
        #     _, temp, _, label_vocab = model.get_words([label_text])
        #     label_words_in_order.append(temp)

        # token_embeddings, candidates_id, candidates = model.get_token_embedding(original_text, words, vocab)
        # label_mask = np.full((len(original_text), model.max_length), 0)
        # for index in range(len(original_text)):
        #     for i,_ in enumerate(candidates[index]):
        #         if i >= model.max_length:
        #             continue
        #         if candidates[index][i] in label_words_in_order[index]:
        #             label_mask[index, i] = 1
        # label_mask = torch.tensor(label_mask).cuda()
        # attention_mask = torch.tensor(np.where(candidates_id == 'None', 0, 1)).cuda() # candidates = torch.tensor(candidates)
        outputs = model.bert(input_ids=input_ids, attention_mask=input_attention_mask, output_hidden_states=True)
        input_token_emb = F.normalize(outputs[0], p=2, dim=-1) # (16, 512, 768)
        align = F.normalize(model.bert(input_ids_label, attention_mask_label).last_hidden_state, p=2, dim=-1)
        attention_logits, all_selected_token_index = model.attention_layer(input_token_emb, align, att_mask_emb=special_tokens_mask, att_mask_proto=attention_mask_label)
        mask = model.AdaptiveMask(attention_logits, all_selected_token_index)  
        selected_mask = mask * input_attention_mask.unsqueeze(-1)

        # p_vecs = model.bert.encode(label, normalize_embeddings=True, convert_to_tensor=True)
        # attention_logits = token_embeddings @ p_vecs.T
        # mask = model.AdaptiveMask(attention_logits)[torch.arange(len(original_text)), :, torch.arange(len(original_text))] #(16, 512)
        # selected_mask = (mask * attention_mask).cpu().detach().numpy().astype(int) #(mask * attention_mask).cpu().detach().numpy().astype(int)
        # selected_mask_ = mask * attention_mask
        # label_mask_ = torch.tensor(label_mask).cuda()
        # proto_acc = torch.sum(selected_mask * label_mask, dim=1) / torch.sum(label_mask, dim=1)
        # proto_acc_mean = torch.mean(proto_acc)
        # proto_recall = torch.sum(selected_mask * label_mask, dim=1) / torch.sum(selected_mask, dim=1)
        # proto_recall_mean = torch.mean(proto_recall)
        # logging.info("acc:", proto_acc_mean)
        # logging.info("recall:", proto_recall_mean)
    #     word_array = np.where(selected_mask == 1, candidates_id, "None")
    #     for i in range(len(original_text)):
    #         selected_words = []
    #         for id_ in word_array[i]:
    #             if id_ != "None":
    #                 selected_words.append(words[int(id_)])
    #         positions = find_keywords(selected_words, original_text[i])
    #         document = highlight_sections_docx(positions, original_text[i], document, label[i])
    #     # selected_sent = [
    #     #         [i for i in word_array[idx, :] if i is not None]
    #     #         for idx in range(len(original_text))
    #     #     ]
    # document.save("/home/bwei2/ProtoTextClassification/_test/" + str(epo_num) + ".docx")
    # return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    incorrect_indices = []
    with torch.no_grad():
        for batch_num,batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # special_tokens_mask = batch['special_tokens_mask'].to(device)
            labels = batch['label'].to(device)
            # offset_mapping = batch['offset_mapping']
            # processed_text = batch['processed_text']
            original_text = batch['original_text']
            # outputs, _, _ = model(input_ids=input_ids, attention_mask=attention_mask,special_tokens_mask=special_tokens_mask, mode="test", original_text = original_text, current_batch_num=batch_num)
            outputs, _, _ = model(input_ids=input_ids, attention_mask=attention_mask, 
                                                current_batch_num=batch_num, original_text = original_text)
            
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    for idx, (pred, actual) in enumerate(zip(predictions, actual_labels)):
        if pred != actual:
            incorrect_indices.append(idx)
    logging.info(incorrect_indices)
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def find_closet_test_sentence(model, data_loader, device):
    model.eval()
    df = pd.read_csv("/home/bwei2/ProtoTextClassification/Data/test_imdb.csv")
    test_texts = df['review'].tolist()
    test_ins_emb = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:

            original_text = batch['original_text']
            test_ins_emb.append(model.bert.encode(original_text, normalize_embeddings=True, convert_to_tensor=False))
    all_test_emb = np.stack(test_ins_emb, axis=0)
    distances = cosine_similarity(model.prototype_vectors, all_test_emb)
    closest_sentences = {}
    top_k = 5
    for proto_idx, proto_distances in enumerate(distances):
        # Get indices of the top_k most similar sentences
        top_k_indices = np.argsort(proto_distances)[-top_k:][::-1]
        # Retrieve the actual sentences
        closest_sentences[proto_idx] = [test_texts[idx] for idx in top_k_indices]
    
    return closest_sentences
    



def train_step(model, val_dataloader, data_loader, optimizer, scheduler, device, accelerator, new_proto, tau=1, train_texts=None):
    model.train()

    Loss = 0
    # aligned_prototype_vectors, sampled_indices = model.align()
    model.train_text = []
    best_result = 0
    real_model = accelerator.unwrap_model(model)
    # 单卡模式不需要 broadcast，同步逻辑可以跳过
    if real_model.prototype_sentence_emb is None:
        real_model.prototype_sentence_emb = real_model.get_proto_sentence_emb().to(accelerator.device)


    # for (batch_num,batch) in enumerate(data_loader):
    #     original_text = batch['original_text']
    #     model.train_text.extend(original_text)
    for (batch_num,batch) in enumerate(data_loader):
        model.batch_num = batch_num
        if batch_num % 5000 == 0 and batch_num >0:
            accuracy, report = evaluate(model, val_dataloader, accelerator.device)
            pnfrl_args = {'bert_model_name': model.args.bert_model_name, 'num_classes': model.args.num_classes, 
                        'prototype_num': model.args.prototype_num, 'batch_size': model.args.batch_size, 
                        'hidden_dim': model.args.hidden_dim, 'max_length': model.args.max_length}
            if accuracy > best_result: 
                best_result = accuracy
                accelerator.save(
                    {'model_state_dict': model.state_dict(), 'pnfrl_args': pnfrl_args},
                    model.args.model_path
                )
                # torch.save({'model_state_dict': model.state_dict(), 'pnfrl_args': pnfrl_args}, model.args.model_path)
            logging.info(f"Batch {batch_num} \n")
            logging.info(f"Validation Accuracy: {accuracy:.4f}\n")
            logging.info(report)
        # else:
        #     LOG=False
        LOG=False
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # special_tokens_mask = batch['special_tokens_mask'].to(device)
        labels = batch['label'].to(device)
        # offset_mapping = batch['offset_mapping'].to(device)
        # processed_text = batch['original_text']
        original_text = batch['original_text']
        # model.train_text.extend(original_text)
        with accelerator.autocast():
            outputs, loss_mu, augmented_loss  = model(input_ids=input_ids, attention_mask=attention_mask, 
                                                    current_batch_num=batch_num, original_text = original_text)
            # span_loss = model.AdaptiveMask.get_loss()
            real_model = accelerator.unwrap_model(model)
            loss = nn.CrossEntropyLoss()(outputs, labels) + 0.1 * loss_mu - 0.001 * real_model.diversity_loss #+ 0.1 * augmented_loss#+ span_loss#+ att_loss.long() + augmented_loss #+ augmented_loss#+ 0.0001 * diversity_loss#+  0.0001 *  locality_loss# + 0.00001 * cLoss #+ 0.001 * eLoss
        # loss += span_loss
        Loss += loss.item()
        logging.info(f"acc : {nn.CrossEntropyLoss()(outputs, labels)}")
        # logging.info(f"mu : {loss_mu}")
        # logging.info(f"div: { model.diversity_loss}")
        # logging.info(f"aug: {augmented_loss}")

        accelerator.backward(loss)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        # logging.info(model.AdaptiveMask.current_val_left)
        # # # if batch_num > 0 and batch_num % 10 == 0:
        # for name, param in model.named_parameters():
        #     # if "current_val" in name:
        #     #     param.grad *= 10000 
        #     logging.info(f"{name} {param.grad}")
        
        scheduler.step()
        # model.AdaptiveMask.clamp_param()

    return Loss

def load_model(path, device_id, log_file=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['pnfrl_args']
    pnfrl = BERTClassifier(bert_model_name=saved_args['bert_model_name'], 
                           num_classes=saved_args['num_classes'], 
                           num_prototype=saved_args['prototype_num'], 
                           batch_size=saved_args['batch_size'], 
                           hidden_dim=saved_args['hidden_dim'], 
                           max_length=saved_args['max_length'])
    pnfrl.load_state_dict(checkpoint['pnfrl_args'])
    return pnfrl

def train_model(args):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device


    if accelerator.is_main_process:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
        if args.log is None:
            logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=log_format)
        else:
            logging.basicConfig(level=logging.INFO, filename=args.log, filemode='w', format=log_format)

    # writer 仅主进程记录
    writer = SummaryWriter(args.folder_path) if accelerator.is_main_process else None


    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:250"
    # rank = args.nr * args.gpus + gpu
    torch.manual_seed(args.r_seed)
    # device_id = args.device_ids[gpu]
    # torch.cuda.set_device(device_id)
    log_file = args.log
    if accelerator.is_main_process:
        writer = SummaryWriter(args.folder_path)
        log_file = args.log
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
        if log_file is None:
            logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
    else:
        writer = None

    adam_epsilon = 1e-8
    num_epochs = args.epoch
    dataset = args.data_set

    model_spec_folder = os.path.join(os.path.join(args.base_folder, args.data_set), args.bert_model_name)

    new_prototype_path = os.path.join(
            model_spec_folder,
            f"{args.data_set}_cluster_{args.prototype_num}"
    ) 

    if not os.path.exists(model_spec_folder):
            os.makedirs(model_spec_folder, exist_ok=True)
            print(f"Created directory: {model_spec_folder}")

    prototype_npy_path = new_prototype_path + "_centers.npy"
    sentence_pool_path = new_prototype_path + "_to_sub_sentence.csv"
   

    gen_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(prototype_npy_path) and os.path.exists(sentence_pool_path):
        print("Prototype vector and sentence pool already exist. Skipping generation.")
    else:
        print("Need to Generate Prototype")
        generate_prototype(args, prototype_npy_path, sentence_pool_path, gen_device)
    train_dataloader, val_dataloader, test_dataloader, tokenizer, train_texts = get_data_loader(dataset,args.dataset_path, args.world_size, 0, args.batch_size, args.max_length, args.bert_model_name,
                                                                                                args.ratio, args.r_seed)

    num_warmup_steps = 0
    num_training_steps = len(train_dataloader)*num_epochs
    logging.info(f"Number of Training Steps : {num_training_steps}")
    model = BERTClassifier(args=args, bert_model_name=args.bert_model_name, num_classes=args.num_classes, 
                           num_prototype=args.prototype_num, 
                           batch_size=args.batch_size, hidden_dim=args.hidden_dim, max_length=args.max_length, 
                           tokenizer=tokenizer).to(device)

    model.tokenizer = tokenizer



    model.args = args
    best_result = 0
    specific_param_left = model.AdaptiveMask.current_val_left
    specific_param_right =  model.AdaptiveMask.current_val_right
    other_params = [param for name, param in model.named_parameters() if param is not specific_param_left and param is not specific_param_right]
    # optimizer = Adam([
    #     {'params': specific_param_left, 'lr': 0.001},  # Learning rate for layer1
    #     {'params': specific_param_right, 'lr': 0.001},
    #     {'params': other_params}, 
    #                    ], lr=args.learning_rate, eps=adam_epsilon)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=adam_epsilon, weight_decay=1e-5 )
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    print("Prepare Acceleration")
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    print("Acceleration loaded")


    

    new_proto = None
    Total_Loss = []
    tau = 0.6
    print("Training Started!")
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        Loss = train_step(model,val_dataloader, train_dataloader, optimizer, scheduler, device, accelerator, new_proto, tau=tau, train_texts=train_texts)
        Total_Loss.append(Loss)


        if accelerator.is_main_process:
            accuracy, report = evaluate(model, val_dataloader, device)

            logging.info(f"Epoch {epoch + 1}/{num_epochs} \n")
            logging.info(f"Validation Accuracy: {accuracy:.4f}\n")
            logging.info(report)
            logging.info(Total_Loss)
            pnfrl_args = {'bert_model_name': args.bert_model_name, 'num_classes': args.num_classes, 
                        'prototype_num': args.prototype_num, 'batch_size': args.batch_size, 
                        'hidden_dim': args.hidden_dim, 'max_length': args.max_length}
            if accuracy > best_result: 
                best_result = accuracy
                accelerator.save({
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'args':args
                }, args.model_path)

    accelerator.wait_for_everyone()
            # torch.save({'model_state_dict': model.state_dict(), 'pnfrl_args': pnfrl_args}, args.model_path)
       



def train_main(args):
    # mp.spawn(train_model, nprocs=args.gpus, args=(args,),join=True)
    train_model(args)


if __name__ == '__main__':
    from args import pnfrl_args
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if '--print_log_path_only' in sys.argv:
        print(pnfrl_args.log)
        sys.exit(0)
    for arg in vars(pnfrl_args):
        print(arg, getattr(pnfrl_args, arg))
    train_main(pnfrl_args)
    # test_model(pnfrl_args)