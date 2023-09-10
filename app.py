# 以下を「app.py」に書き込み
import streamlit as st
import requests
import json
import torch
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertJapaneseTokenizer

import tensorflow as from_tf
import tensorflow as tf

from transformers import BertForSequenceClassification, BertJapaneseTokenizer

#sc_model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=9)
#sc_model.cuda()
#tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
#with zipfile.ZipFile('.zip') as zf:
#    with zf.open('dir_sub/new_file.txt') as f:
#        b = f.read()
url="https://drive.google.com/file/d/1-1ZDrx8LvR4wdD654sBdua2il7l20Q-n/view?usp=sharing"
import urllib.request
#---------------------------------------------------------------------------------------
com=urllib.request.urlopen(url)
ret=com.read()
#com.close()
st.write("モデルを読み込みました1！")
import os

# モデルをダウンロード
#response = requests.get(url)

# 共有リンクからファイルIDを抽出
file_id = "1-1ZDrx8LvR4wdD654sBdua2il7l20Q-n"

# ダウンロードリンクを生成
download_link = f"https://drive.google.com/uc?id={file_id}"

# モデルをダウンロード
response = requests.get(download_link)

# モデルファイルを保存
with open("pytorch_model.bin", "wb") as f:
    f.write(response.content)

st.write("モデルを読み込みました2！")

# モデルを読み込む
loaded_model = BertForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking-uncased",  # 正しいモデル識別子を指定
    num_labels=9  # ラベルの数を指定（必要に応じて調整）
)

# モデルのウェイトを読み込む
loaded_model.load_state_dict(torch.load("pytorch_model.bin"))

# モデルを読み込む
#loaded_model = BertForSequenceClassification.from_pretrained(".", state_dict=response.content)

# Streamlitアプリケーションでモデルを使用
st.write("モデルを読み込みました3！")
com.close()

url3='.'
#loaded_model = BertForSequenceClassification.from_pretrained(url)
#loaded_model = BertForSequenceClassification.from_pretrained(url3)
loaded_model.cuda() 
loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(url3)

st.title("「ニュースの分類」アプリ")
st.write("###### モデル ：Pretrained, Japanese BERT models （東北大学　乾研究室）")
st.write("###### ファインチューニングのコーパス：Livedoorニュースコーパス（ldcc-20140209.tar.gz）")
st.write("###### 分類クラス：「dokujo-tsushin」,「it-life-hack」,「kaden-channel」,「livedoor-homme」,「movie-enter」,「peachy」,「smax」,「sports-watch」,「topic-news」")
st.write("")
dirs=["「dokujo-tsushin」","「it-life-hack」","「kaden-channel」","「livedoor-homme」","「movie-enter」","「peachy」","「smax」","「sports-watch」","「topic-news」"]
text =  st.text_area("ニュースの記事を入力してください。", "首位・阪神は2位・広島との直接対決の初戦を制し、\
今季5度目の6連勝を飾った。チームは75勝44敗4分で貯金は今季最多の31、優勝マジックを「10」に減らした。\
今季19度目先発の村上頌樹（25）は、7回1/3（100球）を投げ、6安打1失点、四死球0の好投。\
自身初の2桁10勝目を挙げた。広島とのゲーム差は今季最大タイの9に。\
ルーキー森下翔太（23）が広島先発の床田に対し、1回2死で3球目のカーブを捉え先制の10号ソロを放った。\
試合前時点で広島戦の打率.325で本塁打2本と、好相性の相手から早速攻撃の口火を切った。\
森下はプロ1年目で2桁本塁打を達成し、球団の右打者としては岡田彰布監督の1980年以来、\
43年ぶりの快挙となった。", height=250)

sample_text = text
sample_text = sample_text.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""}))


max_length = 512
words = loaded_tokenizer.tokenize(sample_text)
word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
word_tensor = torch.tensor([word_ids[:max_length]])  # テンソルに変換

x = word_tensor.cuda()  # GPU対応
y = loaded_model(x)  # 予測
pred = y[0].argmax(-1)  # 最大値のインデックス
st.write("## 予測結果")
st.write("## result:", dirs[pred])

st.write("")

st.write("")



