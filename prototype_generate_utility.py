from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import torch
import joblib  # 用于保存结果
import os
from utils import process
import pandas as pd
from tqdm import tqdm


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def generate_prototype(args, model_spec_folder, prototype_npy_path, sentence_pool_path, device):
    print("Start generate prototype")
    train_data_path = os.path.join(args.dataset_path, "train.csv")
    # print(f"train_data_path: {train_data_path}")
    train_data = pd.read_csv(train_data_path)
    texts = train_data['review'].tolist()
    bert_model = SentenceTransformer(args.bert_model_name)
    
    
    batch_size = 32
    # all_embeddings = []


    with torch.no_grad():
        embedding = bert_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=device
        )

    # print(f"Embeddings shape: {embedding.shape}")

    num_prototypes = args.prototype_num 

    X = embedding.cpu().numpy()

    

    # 如果文件已存在，则跳过 clustering
    if os.path.exists(prototype_npy_path):
        print(f"[SKIP] Prototype file exists: {prototype_npy_path}")
        prototype_vectors = np.load(prototype_npy_path)
    else:
        print(f"[INFO] Start clustering for prototype generation")
    
        kmeans = KMeans(n_clusters=num_prototypes, random_state=args.r_seed, n_init="auto")
        kmeans.fit(X)
        prototype_vectors = kmeans.cluster_centers_
        plot(X, prototype_vectors, num_prototypes)


        np.save(prototype_npy_path, prototype_vectors)

        print(f"Prototype_saved_path: {prototype_npy_path}")
    
    if os.path.exists(sentence_pool_path):
        print(f"[SKIP] Sentence Pool file exists: {sentence_pool_path}")
    else:
        print(f"[INFO] Start Sentence Pool Creation")

        sentence_pool = get_topk_sentences_per_prototype(X, texts, prototype_vectors, k=args.k)
        # all_embeddings.append(embedding)
        # print(f"Embedding Example: {all_embeddings[0]}")
        #  all_embeddings = torch.cat(all_embeddings, dim=0)
        # print(f"Number of train sentences: {len(texts)}")
        # print(f"Shape of embedding: {all_embeddings.shape}")
        save_sentence_pool_to_csv(sentence_pool, sentence_pool_path)
    
    




def get_topk_sentences_per_prototype(X, texts, prototype_vectors, k=40):
    """
    X: [N, D] sentence embeddings
    texts: list of N sentences
    prototype_vectors: [K, D] cluster centers
    return: list of K lists (each list contains top-k sentence strings)
    """
    sentence_pool = []
    for i, center in tqdm(enumerate(prototype_vectors), total=len(prototype_vectors)):
        distances = np.linalg.norm(X - center, axis=1)
        
        # print(distances)
        topk_idx = np.argsort(distances)[-k:]
        # print(topk_idx)
        topk_sentences = [texts[j] for j in topk_idx]
        # print(f"Prototype {i} finished")
        sentence_pool.append(topk_sentences)
        # print('-' * 50)
    return sentence_pool

def save_sentence_pool_to_csv(sentence_pool, out_path):
    df = pd.DataFrame(sentence_pool)
    df.to_csv(out_path, index=False, header=False)
    print(f"Saved sentence pool to {out_path}")


def plot(X, prototype_vectors, num_prototypes):
    X_all = np.vstack([X, prototype_vectors])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_all)

    # 分离句子 embedding 和 prototype
    X_embed_2d = X_2d[:-num_prototypes]
    proto_2d = X_2d[-num_prototypes:]

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(X_embed_2d[:, 0], X_embed_2d[:, 1], s=10, c='lightgray', label='Sentences')
    plt.scatter(proto_2d[:, 0], proto_2d[:, 1], s=100, c='red', marker='x', label='Prototypes')
    plt.legend()
    plt.title("t-SNE Visualization of Sentence Embeddings and Prototype Vectors")
    plt.savefig("prototype_tsne_plot.png", dpi=300)
    plt.show()