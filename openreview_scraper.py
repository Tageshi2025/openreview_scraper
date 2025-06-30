import openreview
import csv
from collections import defaultdict
import os

# 1. 登录 OpenReview API v2
client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username=os.getenv('OPENREVIEW_USERNAME'),
    password=os.getenv('OPENREVIEW_PASSWORD')
)

# 2. 定义会议 & invitation
venue = 'ICML.cc/2025/Conference'
sub_inv = f'{venue}/-/Submission'
review_inv = f'{venue}/-/Paper.*/-/Official_Review'

# 3. 抓取投稿
papers = list(openreview.tools.iterget_notes(client, invitation=sub_inv))
print(f"抓取到 {len(papers)} 篇论文")

# 4. 按 oral / spotlight / poster 分类
by_type = defaultdict(list)
for note in papers:
    venue_info = note.content.get('venue', {}).get('value', '').lower()
    if 'oral' in venue_info:
        by_type['oral'].append(note)
    elif 'spotlight' in venue_info:
        by_type['spotlight'].append(note)
    elif 'poster' in venue_info:
        by_type['poster'].append(note)
    else:
        by_type[venue_info].append(note)

# 5. 写入 CSV
with open('icml2025_openreview.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'paper_id', 'title', 'type', 'keywords', 'abstract'
    ])

    for type_name, plist in by_type.items():
        for p in plist:
            pid = p.id
            title = p.content.get('title', {}).get('value', '').replace('\n', ' ')
            keywords = ",".join(p.content.get('keywords', {}).get('value', []))
            abstract = p.content.get('abstract', {}).get('value', '').replace('\n', ' ')
            writer.writerow([
                pid, title, type_name, keywords, abstract
            ])

print("CSV 文件 icml2025_openreview.csv 已成功生成")
