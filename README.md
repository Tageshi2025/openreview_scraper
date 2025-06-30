# OpenReview Scraper & Hot Topic Analyzer

本项目旨在快速获取最新顶会论文数据，并进行初步的主题聚类分析（效果非常差，建议直接把爬虫结果放到各大主流 LLM 中进行更高效的解读 😓）。

---

## 📌 数据爬取模块

使用 OpenReview 官方 API 自动爬取指定会议的投稿元数据（如标题、摘要、关键词等），结果保存在结构化的 CSV 文件，位于 `data/` 目录中。
示例结果：[ICML2025 爬虫数据结果](https://github.com/Tageshi2025/openreview_scraper/blob/main/data/icml2025_openreview.csv)

## 🔍 主题分析模块

使用 HDBScan、K-means、BERTopic 三种方法进行论文主题聚类分析。

## 🛠 使用教程

1. `git clone` 本仓库  
2. F5 运行 `utils/` 目录下的 `openreview_scraper.py` 实现爬虫  
   （注意修改 OpenReview 账号密码、会议时间和会议名称）  
3. F5 运行 `utils/` 目录下的 `openreview_analyzer.py`，静候 `output/` 中出现主题分析结果  
   （效果很差，不建议使用）
