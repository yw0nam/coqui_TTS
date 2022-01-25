# %%
import os
import pandas as pd
from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
output_path = os.path.join("/home/spow12/codes/TTS/coqui_TTS/", "experiment")

def formatter(root_path, meta_file_train):
    temp = pd.read_csv(os.path.join(root_path, meta_file_train), sep='|')
    return list(zip(temp['normalized_text'], temp['path'], temp['name']))

dataset_config = BaseDatasetConfig(
    name="visual_novel", meta_file_train="train.txt", meta_file_val='val.txt' ,path="/data1/spow12/datas/visual_novel/"
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=False,  # Resample to 22050 Hz. It slows down training. Use `TTS/bin/resample.py` to pre-resample and set this False for faster training.
    do_trim_silence=True,
    trim_db=23.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    preemphasis=0.0,
)

config = Tacotron2Config(  # This is the config that is saved for the future use
    run_name="Tacotron2_visual_novel",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    r=2,
    # gradual_training=[[0, 6, 48], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=False,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
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
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=150,
    print_eval=False,
    mixed_precision=True,
    sort_by_audio_len=True,
    min_seq_len=14800,
    max_seq_len=22050 * 10,  # 44k is the original sampling rate before resampling, corresponds to 10 seconds of audio
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,  # set this to enable multi-sepeaker training
    decoder_ssim_alpha=0.0,  # disable ssim losses that causes NaN for some runs.
    postnet_ssim_alpha=0.0,
    postnet_diff_spec_alpha=0.0,
    decoder_diff_spec_alpha=0.0,
    attention_norm="softmax",
    optimizer="Adam",
    lr_scheduler=None,
    lr=3e-5,
)
ap = AudioProcessor(**config.audio.to_dict())
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

speaker_manager = SpeakerManager()
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
config.num_speakers = speaker_manager.num_speakers

model = Tacotron2(config, speaker_manager)

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
