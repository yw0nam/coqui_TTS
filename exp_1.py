# %%
import pandas as pd
import collections
import os
import random
from multiprocessing import Pool
from typing import Dict, List
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
import cutlet
from TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from TTS.tts.utils.text import pad_with_eos_bos, phoneme_to_sequence, text_to_sequence, sequence_to_text, text2phone
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from glob import glob
import soundfile as sf
import re
# %%
csv = pd.read_csv('../visual_novel/data/data_voice.csv')
# %%
data = csv[csv['normalized_text'] != '………']
data['normalized_text'] = data['normalized_text'].map(lambda x: x.replace('…', '_'))
# %%
i = 2
phonemes = phoneme_to_sequence(
            data['normalized_text'].iloc[i],
            ['basic_cleaners'],
            language='ja-jp',
            enable_eos_bos=False,
            # custom_symbols=custom_symbols,
            tp=  {
                "pad": "_",
                "eos": "~",
                "bos": "^",
                "characters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? ",
                "punctuations": "!'(),-.:;? ",
                "phonemes": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            },
            add_blank=True,
        )
# %%
print(phonemes, '\n',data['normalized_text'].iloc[i])
# %%
pho = np.load("./experiment/phoneme_cache/aya001_019_phoneme.npy")
# %%
sequence_to_text(phonemes)
# %%
text2phone(data['normalized_text'].iloc[i], "ja-jp")
# %%
# %%
len(data['normalized_text'].iloc[0])
# %%
data['normalized_text'].iloc[0]

# %%

# import MeCab
# # %%
# tagger = MeCab.Tagger()
# # %%
# print(tagger.parse(data['normalized_text'].iloc[2]))
# # %%
# data['normalized_text'].iloc[2]

# %%
data['path'] = "/data1/spow12/datas/visual_novel/" + data['game_name'] + '/wav/' + data['voice'].map(lambda x: x.split('|')[0]) + '.wav'
# %%
def wav_file_exist(path):
    try:
        x, sr = sf.read(path)
        return 1
    except RuntimeError:
        return 0
# %%
data = data[data['normalized_text'].map(lambda x: len(x) >= 5)]
data['path_exist'] = data['path'].map(lambda x: wav_file_exist(x))
katsu = cutlet.Cutlet()
katsu.slug(data['normalized_text'].iloc[2])
data['romazi_text'] = data['normalized_text'].map(lambda x: katsu.romaji(x))
data = data.query("path_exist == 1")

# %%
filter_word = ["あああ", "ぃぃぃ"]
re_expr = re.compile("|".join(filter_word))
temp = data.loc[data['normalized_text'].map(lambda x: True if re_expr.match(x) else False)]
to_drop = [5466, 6059, 6368, 17358, 11609, 19021, 19338, 19369, 19957, 31150,
 33729, 34602, 36323, 36678, 36679, 36690, 37138, 37526, 37756, 37843, 37846,
 37934, 47782, 49710, 54865, 61399, 61416]

data = data.drop(index=to_drop)
# %%
data.to_csv("./data/data_voice.path_exist.csv", index=False)
# %%
{
    "\u3042\u3084\u305b": 0,
    "\u30ca\u30c4\u30e1": 1,
    "\u30df\u30ab\u30c9": 2,
    "\u30e0\u30e9\u30b5\u30e1": 3,
    "\u30ec\u30ca": 4,
    "\u4e03\u6d77": 5,
    "\u5343\u54b2": 6,
    "\u5b8f\u4eba": 7,
    "\u5c0f\u6625": 8,
    "\u5e0c": 9,
    "\u5ec9\u592a\u90ce": 10,
    "\u606d\u5e73": 11,
    "\u611b\u8863": 12,
    "\u681e\u90a3": 13,
    "\u6dbc\u97f3": 14,
    "\u7fbd\u6708": 15,
    "\u82a6\u82b1": 16,
    "\u82b3\u4e43": 17,
    "\u8309\u512a": 18,
    "\u8309\u5b50": 19,
    "\u9686\u4e4b\u4ecb": 20
}
# %%
