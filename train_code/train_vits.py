import os, cutlet
import pandas as pd
from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
output_path = os.path.join("/home/spow12/codes/TTS/coqui_TTS/", "experiment")
# %%
def formatter(root_path, meta_file_train):
    temp = pd.read_csv(os.path.join(root_path, meta_file_train), sep='|')
    katsu = cutlet.Cutlet()
    temp['romazi_text'] = temp['normalized_text'].map(lambda x: katsu.slug(x))
    return list(zip(temp['romazi_text'], temp['path'], temp['name']))

dataset_config = BaseDatasetConfig(
    name="visual_novel", meta_file_train="train.txt", meta_file_val='val.txt' ,path="/data1/spow12/datas/visual_novel/"
)
audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=23.0,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
    resample=True,
)
config = VitsConfig(
    audio=audio_config,
    run_name="visual_novel_vits",
    # use_speaker_embedding=True,
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    use_espeak_phonemes=False,
    phoneme_language="ja-jp",
    characters={
        "pad": "_",
        "eos": "~",
        "bos": "^",
        "characters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? ",
        "punctuations": "!'(),-.:;? ",
        "phonemes": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    },
    # phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    # compute_input_seq_cache=True,
    print_step=100,
    print_eval=False,
    mixed_precision=True,
    sort_by_audio_len=True,
    min_seq_len=32 * 256 * 4,
    max_seq_len=1500000,
    output_path=output_path,
    datasets=[dataset_config],
)

ap = AudioProcessor(**config.audio.to_dict())
# %%
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

# # %%
# speaker_manager = SpeakerManager()
# speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
# config.num_speakers = speaker_manager.num_speakers
# %%
model = Vits(config)
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
