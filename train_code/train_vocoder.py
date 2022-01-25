# %%
import os
import pandas as pd
from TTS.config import load_config, register_config
from TTS.trainer import Trainer, TrainingArgs
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseAudioConfig
from TTS.vocoder.datasets.preprocess import load_wav_data, load_wav_feat_data
from TTS.vocoder.models import setup_model
# from TTS.vocoder.configs.melgan_config import MelganConfig
from TTS.vocoder.configs.univnet_config import UnivnetConfig

output_path = os.path.join("./experiment", "visual_novel_melgan_256")

# %%
# load data raw wav files
train_samples = pd.read_csv("/data1/spow12/datas/visual_novel/train.txt", sep='|')['path'].to_list()
eval_samples = pd.read_csv("/data1/spow12/datas/visual_novel/val.txt", sep='|')['path'].to_list()
audio_config = BaseAudioConfig(sample_rate=22050, resample=True, do_trim_silence=True, trim_db=23.0)

# %%
# setup audio processor
config = UnivnetConfig(
    batch_size=256,
    output_path=output_path,
    epochs=1000,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    seq_len=8192
)
ap = AudioProcessor(**config.audio)
# init the model from config
model = setup_model(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(),
    config,
    config.output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
    parse_command_line_args=False,
)
trainer.fit()
# %%
