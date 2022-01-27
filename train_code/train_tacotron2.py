# %%
import os
import cutlet
import pandas as pd
from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager
output_path = os.path.join("/home/spow12/codes/TTS/coqui_TTS/", "experiment")

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
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = FastSpeechConfig(
    run_name="fast_speech_visual_novel",
    audio=audio_config,
    batch_size=16,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    # compute_input_seq_cache=True,
    compute_f0=False,
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
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=100,
    print_eval=False,
    mixed_precision=False,
    sort_by_audio_len=True,
    max_seq_len=600000,
    output_path=output_path,
    datasets=[dataset_config],
)
ap = AudioProcessor(**config.audio.to_dict())
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

# speaker_manager = SpeakerManager()
# speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
# config.num_speakers = speaker_manager.num_speakers

# init the model
model = ForwardTTS(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()