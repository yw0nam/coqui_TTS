# %%
import pandas as pd
import collections
import os
import random
from multiprocessing import Pool
from typing import Dict, List

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
# %%
from glob import glob
import soundfile as sf
# %%
csv = pd.read_csv('./temp.csv')
# %%
names = csv['name'].value_counts().loc[csv['name'].value_counts()>900].index
data = csv.query("name in @names")
data = data[data['normalized_text'] != '………']
# data['normalized_text'] = data['normalized_text'].map(lambda x: x.replace('…', '.'))
# # %%
# i = 2
# phonemes = phoneme_to_sequence(
#             data['normalized_text'].iloc[i],
#             ['phoneme_cleaners'],
#             language='ja-jp',
#             enable_eos_bos=False,
#             # custom_symbols=custom_symbols,
#             # tp=characters,
#             add_blank=True,
#         )
# # %%
# print(phonemes, '\n',data['normalized_text'].iloc[i])
# # %%
# sequence_to_text(phonemes)
# # %%
# text2phone(data['normalized_text'].iloc[i], "ja-jp")
# # %%
# # %%
# len(data['normalized_text'].iloc[0])
# # %%
# data['normalized_text'].iloc[0]

# %%

# import MeCab
# # %%
# tagger = MeCab.Tagger()
# # %%
# print(tagger.parse(data['normalized_text'].iloc[2]))
# # %%
# data['normalized_text'].iloc[2]

# %%
data['path'] = "/data1/spow12/datas/visual_novel/" + data['game_name'] + '/wav/' + data['voice'].map(lambda x: x.split('|')[0]) + '.wav'\
# %%
def wav_file_exist(path):
    try:
        x, sr = sf.read(path)
        return 1
    except RuntimeError:
        return 0
# %%
data = data[data['normalized_text'].map(lambda x: len(x) >= 10)]
data['path_exist'] = data['path'].map(lambda x: wav_file_exist(x))
katsu = cutlet.Cutlet()
katsu.slug(data['normalized_text'].iloc[2])
data['romazi_text'] = data['normalized_text'].map(lambda x: katsu.romaji(x))
data = data.query("path_exist == 1")
# %%
data[['path', 'romazi_text', 'name']].to_csv('/data1/spow12/datas/visual_novel/metadata.txt', sep='|',index=False)
# %%
dataset_config = BaseDatasetConfig(
    name="visual_novel", meta_file_train="metadata.txt", path=os.path.join("/data1/spow12/datas/visual_novel/")
)
def formatter(root_path, meta_file_train):
    temp = pd.read_csv(os.path.join(root_path, meta_file_train), sep='|')
    return list(zip(temp['path'], temp['normalized_text'], temp['name']))

train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

# %%
seq = text_to_sequence(data['romazi_text'].iloc[0],
                 cleaner_names=['basic_cleaners'])
# %%
sequence_to_text(seq)
# %%
# %%
sequence_to_text(t, )
# %%
data.romazi_text.iloc[0]
# %%
