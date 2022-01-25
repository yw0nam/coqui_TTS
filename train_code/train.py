# %%
import os
import pandas as pd
from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
# from TTS.tts.configs.glow_tts_config import GlowTTSConfig
# from TTS.tts.configs.vits_config import VitsConfig
# from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.align_tts_config import AlignTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
# from TTS.tts.models.glow_tts import GlowTTS
# from TTS.tts.models.vits import Vits
# from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.models.align_tts import AlignTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment")
# %%
def formatter(root_path, meta_file_train):
    temp = pd.read_csv(os.path.join(root_path, meta_file_train), sep='|')
    return list(zip(temp['normalized_text'], temp['path'], temp['name']))

dataset_config = BaseDatasetConfig(
    name="visual_novel", meta_file_train="train.txt", meta_file_val='val.txt' ,path="/data1/spow12/datas/visual_novel/"
)
audio_config = BaseAudioConfig(sample_rate=22050, resample=True, trim_db=23.0, do_trim_silence=True)

# config = GlowTTSConfig(
#     batch_size=64,
#     eval_batch_size=16,
#     num_loader_workers=4,
#     num_eval_loader_workers=4,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=1000,
#     text_cleaner="phoneme_cleaners",
#     use_phonemes=True,
#     phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     phoneme_language="ja-jp",
#     print_step=100,
#     print_eval=False,
#     mixed_precision=True,
#     output_path=output_path,
#     datasets=[dataset_config],
#     use_speaker_embedding=True,
#     batch_group_size= 4,
# )
# config = VitsConfig(
#     batch_size=32,
#     eval_batch_size=16,
#     num_loader_workers=4,
#     num_eval_loader_workers=4,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=1000,
#     text_cleaner="phoneme_cleaners",
#     use_phonemes=True,
#     phoneme_language="ja-jp",
#     phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     print_step=100,
#     print_eval=False,
#     mixed_precision=True,
#     output_path=output_path,
#     datasets=[dataset_config],
#     use_speaker_embedding=True,
#     batch_group_size= 2,
# )

config = AlignTTSConfig(
    batch_size=8,
    eval_batch_size=8,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=True,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    phoneme_language="ja-jp",
    characters={
        "pad": "_",
        "eos": "~",
        "bos": "^",
        "characters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? ",
        "punctuations": "!'(),-.:;? ",
        "phonemes": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    },
    print_step=100,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,
    batch_group_size= 8,
)
ap = AudioProcessor(**config.audio.to_dict())
# %%
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

# %%
speaker_manager = SpeakerManager()
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
config.num_speakers = speaker_manager.num_speakers
# %%
# model = GlowTTS(config, speaker_manager)
# model = Vits(config, speaker_manager)
model = AlignTTS(config, speaker_manager)
# %%
trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
# %%
trainer.fit()
