# %%
import pandas as pd
import numpy as np
import cutlet
from collections import Counter, OrderedDict
import re
from sklearn.model_selection import train_test_split
from phonemizer import phonemize
from TTS.tts.utils.text import text2phone
# %%
csv = pd.read_csv('../data/data_voice.path_exist.csv')
names = csv['name'].value_counts().loc[csv['name'].value_counts()>900].index
csv = csv.query("name in @names")
# %%
csv['romazi_text'] = csv['normalized_text'].map(lambda x: text2phone(x, language='ja-jp'))
csv['romazi_text'] = csv['romazi_text'].map(lambda x: x.lower())
csv['romazi_text'] = csv['romazi_text'].map(lambda x: re.sub(r"[^a-zA-Z0-9]","", x))
# %%
chars ="abcdefghijklmnopqrstuvwxyz"
dicts = {}
for i in chars:
    dicts[i] = 0
# %%
csv = csv.reset_index(drop=True)
# %%
def map_fn(x,dicts):
    dic = dicts.copy()
    for char in x:
        dic[char] += (1 / len(x))
    return dic
csv['count'] = csv['romazi_text'].map(lambda x: map_fn(x, dicts))
# %%
count_df = pd.DataFrame(csv['count'].to_list())
need_filtered = pd.DataFrame()
for char in chars:
    temp = csv.loc[count_df.query('%s > 0.4'%(char)).index]
    need_filtered = pd.concat([need_filtered, temp])

ecchi_word = ["じゅぽ", "ちゅば", "ちゅる", "れろれろ", "れろっ", "はああっ", "んんん",
              "んぱぁぁ", "じゅる", "んじゅる", "じゅず"]
re_expr = re.compile("|".join(ecchi_word))

temp = csv.loc[csv['normalized_text'].map(lambda x: True if re_expr.match(x) else False)]
need_filtered = pd.concat([need_filtered, temp])
# %%
need_filtered = need_filtered[~need_filtered.index.duplicated(keep='first')]
idx = need_filtered.index
csv = csv.query("index not in @idx")
# %%
csv['length'] = csv['romazi_text'].map(lambda x: len(x))
# %%
csv_temp = csv.query("length >= 10")
# %%
csv_temp = csv_temp.loc[~csv_temp['normalized_text'].map(lambda x: "%" in x)]
csv_temp = csv_temp.loc[~csv_temp['normalized_text'].map(lambda x: "「" in x)]
csv_temp = csv_temp.loc[~csv_temp['normalized_text'].map(lambda x: ";" in x)]
csv_temp['normalized_text'] = csv_temp['normalized_text'].map(lambda x: x.replace("●", "ん"))
# %%
train, val = train_test_split(csv_temp, test_size=500, random_state=1004, stratify=csv_temp['name'])

# %%
val[['path', 'normalized_text', 'name']].to_csv('/data1/spow12/datas/visual_novel/val.txt', sep='|', index=False)
train[['path', 'normalized_text', 'name']].to_csv('/data1/spow12/datas/visual_novel/train.txt', sep='|', index=False)
# %%
