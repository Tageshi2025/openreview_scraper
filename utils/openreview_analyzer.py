import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import hdbscan
from typing import List
from tqdm import tqdm
import os


INPUT_CSV = './data/icml2025_openreview.csv'
CLUSTERING_METHOD = 'kmeans' # 聚类方法：'hdbscan' 或 'kmeans'
NUM_CLUSTERS = 10 # 聚类数量（仅在 kmeans 下有效）

def load_paper_data(file_path: str) -> pd.DataFrame:
    print("加载论文数据")
    df = pd.read_csv(file_path)
    df["text"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    return df


def compute_text_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    print("提取文本嵌入向量")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def cluster_embeddings(embeddings: np.ndarray, method: str = 'hdbscan', n_clusters: int = 10) -> List[int]:
    print(f"聚类分析方法：{method}")
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(embeddings)
    elif method == 'hdbscan':
        model = hdbscan.HDBSCAN(min_cluster_size=5)
        labels = model.fit_predict(embeddings)
    else:
        raise ValueError("不支持该聚类方法")
    return labels


def reduce_dimensionality(embeddings: np.ndarray) -> np.ndarray:
    print("降维处理")
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    return reduced


def extract_top_keywords(texts: List[str], labels: List[int], top_k: int = 10) -> dict:
    print("提取聚类关键词")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    cluster_keywords = {}
    for label in tqdm(set(labels), desc="关键词提取中"):
        if label == -1:
            continue  # 忽略 HDBSCAN 的噪声
        indices = np.where(labels == label)[0]
        mean_tfidf = np.asarray(tfidf_matrix[indices].mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[::-1][:top_k]
        cluster_keywords[label] = [terms[i] for i in top_indices]
    return cluster_keywords


def visualize_clusters(reduced: np.ndarray, labels: List[int], conference_name: str, output_path: str = None):
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(f"{conference_name} Hot Topic Image", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y")
    if output_path:
        plt.savefig(output_path)
        print(f"聚类图像已保存至{output_path}")
    else:
        plt.show()


def generate_cluster_report(df: pd.DataFrame, labels: List[int], keywords: dict, top_n: int = 3) -> pd.DataFrame:
    df = df.copy()
    df['cluster'] = labels
    report = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        sub = df[df['cluster'] == cluster_id]
        common_titles = sub['title'].head(top_n).tolist()
        report.append({
            '聚类编号': cluster_id,
            '论文数量': len(sub),
            '关键词': ", ".join(keywords.get(cluster_id, [])),
            '代表性标题': " / ".join(common_titles)
        })
    return pd.DataFrame(report)


def extract_conference_name(path: str) -> str:
    base = os.path.basename(path).lower()
    for conf in ['icml', 'cvpr', 'neurips', 'aaai', 'acl', 'emnlp']:
        if conf in base:
            return conf.upper()
    return "unknown conference"


if __name__ == '__main__':
    OUTPUT_CSV = './output/tables/cluster_report.csv'
    OUTPUT_FIG = './output/figures/cluster_visualization.png'

    conference_name = extract_conference_name(INPUT_CSV)
    df = load_paper_data(INPUT_CSV)
    embeddings = compute_text_embeddings(df['text'].tolist())
    labels = cluster_embeddings(embeddings, method=CLUSTERING_METHOD, n_clusters=NUM_CLUSTERS)
    keywords = extract_top_keywords(df['text'].tolist(), labels)
    reduced = reduce_dimensionality(embeddings)
    visualize_clusters(reduced, labels, conference_name, output_path=OUTPUT_FIG)
    summary = generate_cluster_report(df, labels, keywords)
    summary.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"聚类分析完成，报告已保存至：{OUTPUT_CSV}")