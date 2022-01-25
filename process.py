# %%
import pandas as pd
import os
from phonemizer import phonemize

# %%
def formatter(root_path, meta_file_train):
    temp = pd.read_csv(os.path.join(root_path, meta_file_train), sep='|')
    return list(zip(temp['romazi_text'], temp['path'], temp['name']))

# temp = pd.read_csv(os.path.join("/data1/spow12/datas/visual_novel/", "train.txt"), sep='|')
df = pd.read_csv('./temp.csv')
# %%
phonemize(df['normalized_text'].iloc[0], language='ja')
# %%
print(temp['romazi_text'].iloc[0])
pho = phonemize(temp['romazi_text'].iloc[0], language='ja', )
# %%
print(pho)
# %%
