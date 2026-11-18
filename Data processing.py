#数据处理
import pandas as pd
import json
import random
import jieba
import jieba.analyse
import os
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 1.数据加载
df = pd.read_csv('DMSC.csv')
print(f"原始数据总条数：{len(df)}")
print(df.columns.tolist())

#2.数据清洗
# 只保留需要的列
df = df[['Star', 'Comment']]

# 删除空值和重复值
df = df.dropna()
df = df.drop_duplicates(subset=['Comment'])

print(f"清洗后总条数：{len(df)}")

#3.数据标注
# 过滤3星（中性），只保留1-2星（负面）和4-5星（正面）
df = df[df['Star'] != 3]

# 生成标签
df['label'] = df['Star'].apply(lambda x: 1 if x >= 4 else 0)
df = df[['Comment', 'label']]
df.columns = ['text', 'label']

print(f"标注后正面数量：{len(df[df['label'] == 1])}")
print(f"标注后负面数量：{len(df[df['label'] == 0])}")

# 4.数据采样
pos = df[df['label'] == 1].sample(25000, random_state=42)
neg = df[df['label'] == 0].sample(25000, random_state=42)
df_sample = pd.concat([pos, neg]).sample(frac=1, random_state=42)
print(f"采样后总条数：{len(df_sample)}")

# 5.数据集划分
train, temp = train_test_split(df_sample, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"训练集：{len(train)}条")
print(f"验证集：{len(val)}条")
print(f"测试集：{len(test)}条")

# 6.数据增强（EDA）
# 读入同义词库
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "synonyms.json"), "r", encoding="utf-8") as f:
    synonyms = json.load(f)

# 把词库里的词加入jieba
for word in synonyms.keys():
    jieba.add_word(word)


def eda_augment(sentence, p=0.1):
    """对一条评论做EDA数据增强"""
    words = list(jieba.cut(sentence))

    # SR：同义词替换
    words = [
        random.choice(synonyms[w]) if random.random() < p and w in synonyms else w
        for w in words
    ]

    # RI：随机插入
    new_words = []
    for w in words:
        new_words.append(w)
        if random.random() < p and w in synonyms:
            new_words.append(random.choice(synonyms[w]))
    words = new_words

    # RS：随机交换
    for i in range(len(words)):
        if random.random() < p:
            j = random.randint(0, len(words) - 1)
            words[i], words[j] = words[j], words[i]

    # RD：随机删除
    words = [w for w in words if random.random() >= p]

    return ''.join(words)


# 对训练集每条评论生成一条增强数据
print("正在进行数据增强，请稍候...")
augmented = train.copy()
augmented['text'] = augmented['text'].apply(lambda x: eda_augment(str(x)))

# 合并原始训练集和增强数据
train_augmented = pd.concat([train, augmented]).sample(frac=1, random_state=42)
print(f"增强后训练集：{len(train_augmented)}条")

# 7.关键词提取与词云可视化
# 提取正面评论关键词
pos_text = ' '.join(train['text'][train['label'] == 1].tolist())
neg_text = ' '.join(train['text'][train['label'] == 0].tolist())

# 生成正面词云
wordcloud_pos = WordCloud(
    font_path='simhei.ttf',
    width=800,
    height=400,
    background_color='white'
).generate(pos_text)

# 生成负面词云
wordcloud_neg = WordCloud(
    font_path='simhei.ttf',
    width=800,
    height=400,
    background_color='white'
).generate(neg_text)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(wordcloud_pos, interpolation='bilinear')
axes[0].set_title('正面评论关键词', fontsize=16)
axes[0].axis('off')

axes[1].imshow(wordcloud_neg, interpolation='bilinear')
axes[1].set_title('负面评论关键词', fontsize=16)
axes[1].axis('off')

plt.savefig('wordcloud.png', dpi=150, bbox_inches='tight')
plt.show()
print("词云图已保存为 wordcloud.png")

# 8.保存处理好的数据集
train_augmented.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)
print("数据集已保存")