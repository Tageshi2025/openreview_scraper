import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# ========== 参数配置 ==========
INPUT_CSV = './data/icml2025_openreview.csv'
PREFIX = os.path.basename(INPUT_CSV).split('_')[0]
OUTPUT_TABLE = f'./output/tables/{PREFIX}_bertopic_summary.csv'
OUTPUT_WORDCLOUD_DIR = f'./output/figures/{PREFIX}'
os.makedirs(OUTPUT_WORDCLOUD_DIR, exist_ok=True)
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


# ========== 加载数据 ==========
def load_data(csv_path: str) -> pd.DataFrame:
    print("加载论文数据中...")
    df = pd.read_csv(csv_path)
    df["text"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    return df

# ========== 保存词云图 ==========
def save_topic_wordclouds(model: BERTopic, output_dir: str, max_words: int = 30):
    os.makedirs(output_dir, exist_ok=True)
    for topic_id in model.get_topics().keys():
        if topic_id == -1:
            continue
        topic_words = dict(model.get_topic(topic_id))
        wc = WordCloud(width=800, height=600, background_color='white', max_words=max_words)
        wc.generate_from_frequencies(topic_words)
        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {topic_id}", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{PREFIX}_topic_{topic_id}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"保存词云图：{save_path}")

# ========== 执行流程 ==========
if __name__ == '__main__':
    print("=== 开始 BERTopic 热点分析 ===")
    df = load_data(INPUT_CSV)
    print("加载嵌入模型...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("初始化主题模型...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        verbose=True
    )
    print("训练 BERTopic 模型中...")
    topics, probs = topic_model.fit_transform(df['text'].tolist())

    save_topic_wordclouds(topic_model, OUTPUT_WORDCLOUD_DIR)
    topic_info = topic_model.get_topic_info()

    os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)
    topic_info.to_csv(OUTPUT_TABLE, index=False, encoding='utf-8-sig')
    print(f"\n分析完成，结果已保存至：{OUTPUT_TABLE}")
