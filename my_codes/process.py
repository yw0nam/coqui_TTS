# %%
from TTS.tts.datasets.dataset import TTSDataset
import numpy as np
import os
import pandas as pd
from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text import pad_with_eos_bos, phoneme_to_sequence, text_to_sequence, sequence_to_text, text2phone
# %%
output_path = os.path.join("/home/spow12/codes/TTS/coqui_TTS/", "experiment")
# %%
def formatter(root_path, meta_file_train):
    temp = pd.read_csv(os.path.join(root_path, meta_file_train), sep='|')
    return list(zip(temp['normalized_text'], temp['path'], temp['name']))
# %%


# %%
dataset_config = BaseDatasetConfig(
    name="visual_novel", meta_file_train="train.txt", meta_file_val='val.txt' ,path="/data1/spow12/datas/visual_novel/"
)
audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=23.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)
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
    text_cleaner=["phoneme_cleaners"],
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
    use_speaker_embedding=True,
)
ap = AudioProcessor(**config.audio.to_dict())

speaker_id_mapping = {
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
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

# %%
speaker_manager = SpeakerManager()
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
config.num_speakers = speaker_manager.num_speakers
# %%
# class TTSDataset_for_jp(TTSDataset):
#     def _generate_and_cache_phoneme_sequence(
#         text, cache_path, cleaners, language, custom_symbols, characters, add_blank
#     ):
#         """generate a phoneme sequence from text.
#         since the usage is for subsequent caching, we never add bos and
#         eos chars here. Instead we add those dynamically later; based on the
#         config option."""
#         phonemes = phoneme_to_sequence(
#             text,
#             ['phoneme_cleaners'],
#             language='ja-jp',
#             enable_eos_bos=False,
#             custom_symbols=custom_symbols,
#             tp=  {
#                 "pad": "_",
#                 "eos": "~",
#                 "bos": "^",
#                 "characters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? ",
#                 "punctuations": "!'(),-.:;? ",
#                 "phonemes": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
#             },
#             add_blank=True,
#         )
#         phonemes = np.asarray(phonemes, dtype=np.int32)
#         np.save(cache_path, phonemes)
#         return phonemes
# %%
dataset = TTSDataset(
    outputs_per_step=config.r if "r" in config else 1,
    text_cleaner="phoneme_cleaners",
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
    speaker_id_mapping=speaker_id_mapping,
    d_vector_mapping=None
)
# %%
data = dataset.load_data(2550)
# %%
train_samples[0]
# %%
temp = np.load("../experiment/phoneme_cache/kan006_024_phoneme.npy")
# %%
# %%
sequence_to_text(temp)
# %%
i = 2
phonemes = phoneme_to_sequence(
            data['raw_text'],
            ['phoneme_cleaners'],
            language='ja-jp',
            enable_eos_bos=False,
            # custom_symbols=custom_symbols,
            # tp=  {
            #     "pad": "_",
            #     "eos": "~",
            #     "bos": "^",
            #     "characters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? ",
            #     "punctuations": "!'(),-.:;? ",
            #     "phonemes": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            # },
            add_blank=False,
        )
# %%
sequence_to_text(phonemes)

# %%
np.array(phonemes)
# %%
dataset.phoneme_language
# %%
dataset.custom_symbols
# %%
