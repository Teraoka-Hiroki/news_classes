# 以下を「app.py」に書き込み
import streamlit as st
import torch
from transformers import BertForMaskedLM
from transformers import BertTokenizer

from transformers import BertForSequenceClassification, BertJapaneseTokenizer

#sc_model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=9)
#sc_model.cuda()
#tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

news_path = "/content/drive/MyDrive/Mago_Serise/mago120(news_clasification)/"
loaded_model = BertForSequenceClassification.from_pretrained(news_path)
loaded_model.cuda()
loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(news_path)

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



