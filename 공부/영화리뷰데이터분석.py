import os
import re

import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils

data_set = tf.keras.utils.get_file(
      fname="imdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True) # 압축파일, 해당url에서 데이터다운로드, 압축해제True

def directory_data(directory):
    data = dict()
    data["review"] = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), "r") as file:
            data["review"].append(file.read())
    
    # dict을 판다스의 데이터프레임으로 변환함
    return pd.DataFrame.from_dict(data)

# pos폴더에 접근할지 neg폴더에 접근할지 
def data(directory):
    pos_df = directory_data(os.path.join(directory, "pos"))
    neg_df = directory_data(os.path.join(directory, "neg"))
    pos_df["sentiment"] = 1
    neg_df["sentiment"] = 0

    return pd.concat([pos_df, neg_df])

train_df = data(os.path.join(os.path.dirname(data_set), "aclImdb", "train"))
test_df = data(os.path.join(os.path.dirname(data_set), "aclImdb", "test"))

# 데이터프레임 결과 확인
train_df.head() 

# pandas의 데이터프레임으로부터 문장 리스트를 가져옴
reviews = list(train_df['review'])

# 문자열 문장 리스트를 토큰화(tokenizing)
tokenized_reviews = [r.split() for r in reviews]
# 토크나이징된 리스트에 대한 각 길이를 저장
review_len_by_token = [len(t) for t in tokenized_reviews]
# 토크나이징된 것을 붙여서 음절의 길이를 저장
review_len_by_syllable = [len(s.replace(' ','')) for s in reviews]


import matplotlib.pyplot as plt

#그래프에 대한 이미지 크기 선언
# figsize: (가로, 세로) 형태의 튜플로 입력한다
plt.figure(figsize=(12, 5))
# 히스토그램 선언
# bins: 히스토그램 값에 대한 버킷 범위
# alpha: 그래프 색상 투명도
# color: 그래프 색상
# label: 그래프에 대한 라벨
plt.hist(review_len_by)
plt.hist(review_len_by_token, bins=50, alpha=0.5, color= 'r', label='word')
plt.hist(review_len_by_syllable, bins=50, alpha=0.5, color='b', label='alphabet')
plt.yscale('log', nonposy='clip')
# 그래프 제목
plt.title('Review Length Histogram')
# 그래프 x 축 라벨
plt.xlabel('Review Length')
# 그래프 y 축 라벨
plt.ylabel('Number of Reviews')


import numpy as np

print('문장 최대길이: {}'.format(np.max(review_len_by_token)))
print('문장 최소길이: {}'.format(np.min(review_len_by_token)))
print('문장 평균길이: {:.2f}'.format(np.mean(review_len_by_token)))
print('문장 길이 표준편차: {:.2f}'.format(np.std(review_len_by_token)))
print('문장 중간길이: {}'.format(np.median(review_len_by_token)))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('제 1 사분위 길이: {}'.format(np.percentile(review_len_by_token, 25)))
print('제 3 사분위 길이: {}'.format(np.percentile(review_len_by_token, 75)))

plt.figure(figsize=(12, 5))
# 박스플롯 생성
# 첫번째 파라메터: 여러 분포에 대한 데이터 리스트를 입력
# labels: 입력한 데이터에 대한 라벨
# showmeans: 평균값을 마크함

plt.boxplot([review_len_by_token],
             labels=['문장 내 단어 수에 대한 히스토그램'],
             showmeans=True)

plt.figure(figsize=(12, 5))
plt.boxplot([review_len_by_eumjeol],
             labels=['문장 내 알파벳 개수에 대한 히스토그램'], 
             showmeans=True)


# 워드클라우드 시각화 -> 등장횟수와 단어의 크기가 비례. 이상치가 있다면 제거하며 전처리.
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline

wordcloud = WordCloud(stopwords = STOPWORDS, background_color = 'black', width = 800, height = 600).generate(' '.join(train_df['review']))

plt.figure(figsize = (15, 10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# 긍정부정 -> 긍정 부정으로 나누었는데 12000개로 같음. 데이터의 균형이 좋다. -> 균형이 안좋은 경우는 어떻게 할까?
import seaborn as sns
import matplotlib.pyplot as plt

sentiment = train_df['sentiment'].value_counts()
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(train_df['sentiment'])