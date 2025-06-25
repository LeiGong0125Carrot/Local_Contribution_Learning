import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertConfig, BertModel
import os
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import torch.nn.functional as F
from torch import where, rand_like, zeros_like, log, sigmoid, tanh
from torch.nn import Module
from transformers import AutoTokenizer, AutoModel, AutoConfig
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, BertModel
from transformers import  DistilBertModel, DistilBertTokenizerFast
from torch.autograd import Variable
import logging
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold
from utils import *
import math
# from model_ours import *
from sklearn.feature_extraction.text import CountVectorizer
# from gpu_mem_track import MemTracker
# import GPUtil
import gc
from sklearn.metrics.pairwise import cosine_similarity
# from _highlight import highlight_document
import regex as re # important - not using the usual re module
import torch.optim as optim
import torch.distributions as dist
# import docx
from scipy.special import softmax
Representative_Instance = []
import pickle
import json
from AdaptiveMask import *
# from CrossAttention import *
EPSILON = np.finfo(np.float32).tiny
# from MultiHeadSelfAttention import *
from DPGMM import *
from transformers import AutoModel, AutoTokenizer



def tail_inclusive_analyzer(doc, k=5, pad_token="<PAD>"):
    tokens = doc.split()                     # 或用自己的 tokenizer
    L = len(tokens)
    for i in range(L):
        # 取 i 开头的 k 个 token；若不足 k 就右侧 PAD
        span = tokens[i:i+k]
        if len(span) < k:
            span += [pad_token] * (k - len(span))
        yield " ".join(span)



class BERTClassifier(nn.Module):
    def __init__(self, args, bert_model_name, num_classes, num_prototype, batch_size, hidden_dim, max_length, distributed=False, tokenizer=None):
        super(BERTClassifier, self).__init__()
        # self.bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #all-MiniLM-L6-va
        
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # print("max position embedding length:", self.bert.config.max_position_embeddings)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        
        self.args = args
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototype
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_spec_folder = os.path.join(os.path.join(args.base_folder, args.data_set), args.bert_model_name)
        self.sentence_pool = pd.read_csv(os.path.join(self.model_spec_folder, str(args.data_set)+"_cluster_"+str(self.num_prototypes)+"_to_sub_sentence.csv"), index_col=0, header=None).to_numpy()
        
        self.prototype_sentence_emb = None
        
        self.prototype_vectors = nn.Parameter(torch.tensor(np.load(os.path.join(self.model_spec_folder, str(args.data_set)+"_cluster_"+str(self.num_prototypes)+"_centers.npy"))),requires_grad=True)
        # self.prototype_vectors = nn.Parameter(self.ran_sent_init(),requires_grad=True)
        # self.prototype_vectors = nn.Parameter(torch.randn(self.num_prototypes, self.hidden_dim), requires_grad=True)
        self.AdaptiveMask = AdaptiveMask()
        self.AdaptiveMask.max_length = self.max_length
        self.fc = nn.Linear(self.num_prototypes, num_classes)
        self.epsilon = 1e-4
        # self.gumbel = GumbelSigmoid()
        self.samples = None
        self.keyphrase_ngram_range = (args.window_size, args.window_size)
        self.count = CountVectorizer(
            # ngram_range=self.keyphrase_ngram_range,
            # stop_words=[],
            # token_pattern=r'(?u)\b\w+\b',
            analyzer=lambda d: tail_inclusive_analyzer(d, k=args.window_size)
            )
        self.num_gau_components = self.args.gaussian_num
        self.mdn = MixtureDensityNetwork(n_input=self.max_length, n_hidden=self.max_length, n_components=self.num_gau_components)
        self.aligned_prototype_vectors = None
    def pairwise_cosine(self, tensor_a, tensor_b):
        tensor_a_normalized = F.normalize(tensor_a, p=2, dim=-1)  # Shape: (16, 256, 384)
        tensor_b_normalized = F.normalize(tensor_b, p=2, dim=-1)  # Shape: (8, 384)
        tensor_b_normalized = tensor_b_normalized.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 8, 384)
        similarity = torch.matmul(tensor_a_normalized.unsqueeze(2), tensor_b_normalized.transpose(-1, -2))  # Shape: (16, 256, 8, 1)
        similarity = similarity.squeeze()
        return similarity
    
    def get_start_point(self, train_text, df, words, mode="train", batch_num=None, prototype_vec=None):
        # Get word embeddings for the provided words
        # print(f"Number of words embedding: {len(words)}")
        encoding = self.tokenizer(
            words,
            padding=True,
            truncation=True,
            max_length=self.args.window_size,
            return_tensors='pt'
        ).to(self.bert.device)
        
        
        outputs = self.bert(**encoding)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        word_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
        # print(f"Words Embedding dimension: {word_embeddings.shape}")
        # exit(0)
        # word_embeddings = self.bert(words, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)

        top_n = 1
        start_points_index = []
        max_distances = []
        padded_emb = []
        padded_candidates_words = []
        # cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        # Iterate through each instance in the training text
        for index, instance in enumerate(train_text):
            # Get candidate indices and corresponding words
            candidate_indices = df[index].nonzero()[1]  # Non-zero indices from dataframe

            candidates = [words[i] for i in candidate_indices]  # Words corresponding to non-zero indices
            # Get the corresponding embeddings for the candidates
            candidate_embeddings = word_embeddings[candidate_indices]
            
            k = candidate_embeddings.shape[0]
            if k < self.max_length:
                # Padding the embeddings with zeros if there are fewer than 512 candidates
                padding_size = self.max_length - k
                candidate_embeddings_ = torch.cat([candidate_embeddings, torch.zeros(padding_size, self.hidden_dim).cuda()], dim=0)
                
                # Padding the candidate words list with "None"
                candidates += ["None"] * padding_size
            else:
                # If there are more than 512 candidates, truncate both embeddings and candidates
                candidate_embeddings_ = candidate_embeddings[:self.max_length, :]
                candidates = candidates[:self.max_length]
            padded_emb.append(candidate_embeddings_)
            padded_candidates_words.append(candidates)
        padded_emb = torch.stack(padded_emb, dim=0)
        
        distances = self.pairwise_cosine(padded_emb, self.prototype_vectors.detach()) # 16, 256, 8
        return padded_emb, padded_candidates_words, distances


    
    def forward(self, input_ids=None, attention_mask=None,
                mode="train", current_batch_num=None, original_text=None):
        
        
        
        # self.bert.eval()
        if current_batch_num % 200 == 0 and mode=="train" and current_batch_num >0:
            aligned_prototype_vectors = self.align()
        else: 
            aligned_prototype_vectors = self.prototype_vectors
        torch.cuda.empty_cache()
        # x = self.bert._first_module().auto_model.embeddings(input_ids, attention_mask)

        new_input_ids = input_ids.unsqueeze(1).expand(-1,  self.num_prototypes, -1).reshape(-1, self.max_length)
        label_mask = input_ids.unsqueeze(1).expand(-1,  self.num_prototypes, -1)
        # example = ["I loves panda jiaying is a piggy she is fat and stupid"]
        # original_text = example
        words, words_in_order, df, vocab = self.get_words(original_text = original_text)
         


        candidates_embeddings, candidate_words, chunk_similarity = self.get_start_point(original_text, df, words_in_order, mode=mode, batch_num=current_batch_num, prototype_vec=aligned_prototype_vectors) #(16, 512, 768)
        
        # print(f"Chunk Similarity dimension: {chunk_similarity.shape}")
        # print(f"chunk similarity dimension: {chunk_similarity.shape}")
        chunk_similarity = chunk_similarity.permute(0, 2, 1).reshape(-1, self.max_length)
        # print(f"Reshaped Chunk Similarity dimension: {chunk_similarity.shape}")
        
        pi, mu, sigma = self.mdn(chunk_similarity)#.reshape(input_ids.shape[0], self.num_prototypes, -1)

        mu_label = torch.topk(chunk_similarity, pi.shape[1], dim=1).indices

        loss_mu = self.mdn.loss(pi, mu, sigma , mu_label)
        mask = self.AdaptiveMask(mu, sigma, pi, batch_size=input_ids.shape[0], num_prototypes=self.num_prototypes).reshape(input_ids.shape[0], self.num_prototypes, -1)
        # print(f"Mask dimension: {mask.shape}")    

        # x = self.bert._first_module().auto_model(input_ids, attention_mask, output_hidden_states=False).last_hidden_state#self.bert.encode(original_text,  normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False, output_value="token_embeddings")
        


        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False).last_hidden_state


        Z_prime = self.mean_pooling(x, mask.permute(0,2, 1)).permute(0,2,1)#torch.sum(x * mask.unsqueeze(-1), dim=2) / torch.clamp(mask.unsqueeze(-1).sum(2), min=1e-9) #self.mean_pooling(x, mask)#torch.stack(outputs_, dim=1).squeeze()#.permute(1, 0, 2, 3) #(16, 8, 512, 768)

        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
       
        similarity = cos(Z_prime, aligned_prototype_vectors.unsqueeze(0)) #torch.exp(1/self.temperature_cosine * cos(z_prime, F.normalize(self.prototype_vectors, p=2,dim=1)))

        augmented_loss = 0#torch.clamp(0.2 - similarity, min=0).mean()
        logits = self.fc(similarity)
        self.diversity_loss = self._diversity_term(self.prototype_vectors)#self.diversity(mask)
        return logits, loss_mu, augmented_loss

    def mean_pooling_sentence(self, model_output, attention_mask):
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def mean_pooling(self, token_embeddings, attention_mask):

        input_mask_expanded = attention_mask.unsqueeze(2).float()
        return torch.mean(token_embeddings.unsqueeze(-1) * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def get_words(self, original_text):
        self.count.fit(original_text)
        words = self.count.get_feature_names_out()
        words = [i for i in words]
        words_in_order = list(self.count.vocabulary_.keys())#[::keyphrase_ngram_range[0]]#count.get_feature_names_out()
        vocab = self.count.vocabulary_

        df = self.count.transform(original_text)
        return words, words_in_order, df, vocab
    
    def _diversity_term(self, x, d="euclidean", eps=1e-9):
        if d == "euclidean":
            # euclidean distance
            D = torch.cdist(x, x, 2)
            Rd = torch.relu(-D + 0.5)
            # logging.info(D)
            zero_diag = torch.ones_like(Rd, device=Rd.device) - torch.eye(
                x.shape[-2], device=Rd.device
            )
            return ((Rd * zero_diag)).sum() / 2.0
        elif d == "cosine":
            # cosine distance
            x_n = x.norm(dim=-1, keepdim=True)
            x_norm = x / torch.clamp(x_n, min=eps)
            D = 1 - torch.matmul(x_norm, x_norm.transpose(-1, -2))
            zero_diag = torch.ones_like(D, device=D.device) - torch.eye(
                x.shape[-2], device=D.device
            )
            return (D * zero_diag).sum() / 2.0
        else:
            raise NotImplementedError


    '''def get_proto_sentence_emb(self):
        print(f"Sentence pool: {self.sentence_pool}")
        exit(0)
        result = torch.stack([self.bert.encode( self.sentence_pool[i,:], normalize_embeddings=True, convert_to_tensor=True).unsqueeze(0) for i in range(self.num_prototypes)]).squeeze()
        return result
    
        for i in range(self.num_prototypes):
            batch = self.tokenizer(list(self.sentence_pool[i, :]), return_tensors='pt',
                           padding='max_length', truncation=True, max_length=self.max_length)
            batch = {k: v.cuda() for k, v in batch.items()}
            
            outputs = self.bert(**batch)
            sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, H]
            cls_embedding = F.normalize(cls_embedding, p=2, dim=1) '''
    
    def get_proto_sentence_emb(self):
    
        # print(f"Start Proto Get!")
        all_embeddings = []
        for i in range(self.num_prototypes):
            # 取出当前 prototype 对应的句子池
            # print(f"{i}th prototype")
            sentences = list(self.sentence_pool[i, :])
            
            # 编码成 input_ids, attention_mask
            batch = self.tokenizer(sentences, return_tensors='pt',
                                padding=True, truncation=True,
                                max_length=self.max_length)
            batch = {k: v for k, v in batch.items()}
            
           
            input_ids = batch['input_ids'].to(next(self.bert.parameters()).device)
            attention_mask = batch['attention_mask'].to(next(self.bert.parameters()).device)

            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].unsqueeze(0)   # [B, H]
                
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)  # L2 normalization
            # 对该 prototype 对应的多个句子求平均
            
            all_embeddings.append(cls_embeddings)
            # print(f"{i}th prototype finished")
        
        return torch.stack(all_embeddings).squeeze()  # [num_prototypes, H]
    
    def get_token_embedding(self, original_text, words, vocab, emb = True):
        candidates = []
        for i in range(len(original_text)):
            _, words_in_order, _, _= self.get_words([original_text[i]])
            candidates.append(words_in_order)
        candidates_id = np.full((len(original_text), self.max_length), "None")
        for index in range(len(original_text)):
            for i,_ in enumerate(candidates[index]):
                if i >= self.max_length:
                    continue
                candidates_id[index, i] = vocab[candidates[index][i]]
        if emb: 
            words_emb = self.bert.encode(words, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar = False)
            token_embeddings = torch.zeros((len(original_text), self.max_length, 768)).cuda()
            for idx in range(len(original_text)):
                for pos, word_id in enumerate(candidates_id[idx, :]):
                    if word_id != "None":
                        token_embeddings[idx, pos, :] = words_emb[int(word_id), :]
        else:
            token_embeddings = None
        return token_embeddings, candidates_id, candidates


    def get_word_emb(self, words):
        word_embeddings = self.bert.encode(words, normalize_embeddings=True, convert_to_tensor=False, show_progress_bar = False)
        return word_embeddings

    
    '''def get_all_start_point(self, train_text, batch_num):
        words, words_in_order, df, vocab = self.get_words(original_text = train_text)

        word_embeddings = self.get_word_emb(words)

        top_n = 1
        self.start_point_dict = {}
        for prototype_id in range(self.num_prototypes):
            for proto_sent_id in range(self.prototype_sentence_emb.shape[1]):
                self.start_point_dict[(prototype_id, proto_sent_id)] = []
                doc_embeddings = self.prototype_sentence_emb[prototype_id ,proto_sent_id, :].squeeze().detach().cpu().numpy()
                for index, instance in enumerate(train_text):
                    candidate_indices = df[index].nonzero()[1]
                    candidates = [words[index] for index in candidate_indices]
                    candidate_embeddings = word_embeddings[candidate_indices]
                    distances = cosine_similarity([doc_embeddings], candidate_embeddings)
                    keywords = [
                        candidates[index]
                        for index in distances.argsort()[0][-top_n:]
                    ][::-1]
                    encoding = self.tokenizer(keywords,  add_special_tokens=False)
                    self.start_point_dict[(prototype_id, proto_sent_id)] = encoding['input_ids']
                    with open(start_point_batch_file, 'wb') as file_:
                        pickle.dump(self.start_point_dict, file_) '''


    '''def ran_sent_init(self, ):
        df = pd.read_csv("/scratch/nkw3mr/intepre_clinical_long_doc/ProtoLens/Datasets/" + str(self.args.data_set) + "/"+str(self.args.data_set)+"_train_sub_sentences.csv")["review"].tolist()  # Replace "xx.csv" with your actual file path
        # Randomly select 10 indices
        random_indices = np.random.choice(len(df), size=self.num_prototypes, replace=False)
        # Get the elements at these random indices
        selected_elements = [df[i] for i in random_indices]
        ran_embeddings = self.bert.encode(selected_elements, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        return ran_embeddings'''
    def locality(self, mask, sent_mask=None):
        mask = torch.permute(mask, (0, 2, 1))
        threshold = 0.2 * sent_mask.unsqueeze(1).sum(2)
        if sent_mask is not None:
            return torch.relu(((mask.sum(2) - threshold)  / sent_mask.unsqueeze(1).sum(2)).sum())
        else:
            return mask.mean(2).sum()


    def get_mask(self, attention_logits, mode="train", log=False, mask=None, return_soft=False):
        if mode == "train":
            mask = self.gumbel(attention_logits,Log=log, mask=mask, return_soft=return_soft)
        elif mode == "test":
            mask = self.gumbel(attention_logits, mask=mask, return_soft=return_soft)
        return mask

    def align(self, ):
        prototypes = self.prototype_vectors 
        total_emb = None

        prototype_sentence_emb = self.prototype_sentence_emb.to(self.prototype_vectors.device)#total_emb#self.prototype_sentence_emb 

        # print(prototype_sentence_emb.shape) 
        # prototypes_normalized = F.normalize(prototypes, dim=-1)  # Shape: (8, 768)
        # candidates_normalized = F.normalize(prototype_sentence_emb, dim=-1)  # Shape: (8, 40, 768)

        cosine_sim = torch.einsum('ij,ikj->ik', prototypes, prototype_sentence_emb)  # Shape: (8, 40)
        # cosine_sim = torch.matmul(F.normalize(prototypes), F.normalize(prototype_sentence_emb).T)
        # Select the indices of the top 3 most similar candidates for each prototype
        topk_values, topk_indices = torch.topk(cosine_sim, k=1, dim=-1)  # Shape of topk_indices: (8, 3)
        logging.info(f"align result:")
        for prototype_idx, sentence_indices in enumerate(topk_indices):
            logging.info(f"Prototype {prototype_idx + 1}:")
            for rank, idx in enumerate(sentence_indices):
                # logging.info(f"  Most similar sentence {rank + 1}: {self.train_text[idx.item()]}")
                logging.info(f"  Most similar sentence {rank + 1}: {self.sentence_pool[prototype_idx, idx]}")

        selected_candidates =  torch.mean(torch.stack([prototype_sentence_emb[i, idx] for i, idx in enumerate(topk_indices)]), dim=1)  # Shape: (8, 768)


        threshold = 0.5
        softness = 10.0  # Controls the sharpness of the transition; higher = more like a hard threshold

        # Compute the pairwise Euclidean distance between corresponding vectors in a and b
        distances = torch.norm(prototypes - selected_candidates, dim=1)

        # Compute the "soft mask" that smoothly blends between the two cases
        weights = torch.sigmoid(softness * (distances - threshold))

        # Create a "soft" move of a toward b based on the threshold distance
        direction = (selected_candidates - prototypes) / (distances.unsqueeze(1) + 1e-8)  # Unit vector, avoid division by zero
        move_toward_b = prototypes + direction * threshold  # Moving a towards b by the threshold distance

        # The final output is a smooth combination of b and the move toward b
        prototype_updated = weights.unsqueeze(1) * move_toward_b + (1 - weights.unsqueeze(1)) * selected_candidates
        
        aligned_prototype_vectors = (selected_candidates - self.prototype_vectors).detach() + self.prototype_vectors
        return aligned_prototype_vectors#, sampled_indices

