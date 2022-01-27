# %%
from TTS.tts.datasets.dataset import TTSDataset
import numpy as np
import os
import pandas as pd
from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text import pad_with_eos_bos, phoneme_to_sequence, text_to_sequence, sequence_to_text, text2phone
from TTS.tts.configs.fast_speech_config import FastSpeechConfig

output_path = os.path.join("/home/spow12/codes/TTS/coqui_TTS/", "experiment")
# %%
dataset_config = BaseDatasetConfig(
    name="kokoro", meta_file_train="metadata.csv", path="./../../Kokoro-Speech-Dataset/output/"
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
    resample=False,
)
# %%
config = FastSpeechConfig(
    run_name="FastSpeech_visual_novel",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    # compute_input_seq_cache=True,
    # compute_f0=True,
    # f0_cache_path=os.path.join(output_path, "f0_cache"),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
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
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    sort_by_audio_len=True,
    max_seq_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=False,
)
# %%
ap = AudioProcessor(**config.audio.to_dict())
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# %%
# speaker_manager = SpeakerManager()
# speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
# config.num_speakers = speaker_manager.num_speakers
dataset = TTSDataset(
    outputs_per_step=config.r if "r" in config else 1,
    text_cleaner=config.text_cleaner,
    compute_linear_spec=config.model.lower() == False,
    compute_f0=config.get("compute_f0", False),
    f0_cache_path=config.get("f0_cache_path", None),
    meta_data=train_samples,
    ap=ap,
    characters=config.characters,
    add_blank=config["add_blank"],
    return_wav=config.return_wav if "return_wav" in config else False,
    batch_group_size=0,
    min_seq_len=config.min_seq_len,
    max_seq_len=config.max_seq_len,
    phoneme_cache_path=config.phoneme_cache_path,
    use_phonemes=config.use_phonemes,
    phoneme_language=config.phoneme_language,
    enable_eos_bos=config.enable_eos_bos_chars,
    use_noise_augment=False,
    verbose=False,
    speaker_id_mapping=None,
    d_vector_mapping=None
)
# %%
data = dataset.load_data(2550)
# %%
data
# %%
sequence_to_text(data['text'])
# %%
config
# %%
