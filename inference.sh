python ./TTS/bin/synthesize.py --text "行けばバイト代も出るわよ。もちろん交通費も渡す" \
    --speakers_file_path "experiment/coqui_tts-January-21-2022_05+16AM-7f1a2378/speakers.json"\
    --model_path "experiment/coqui_tts-January-21-2022_05+16AM-7f1a2378/best_model.pth.tar" \
    --config_path "experiment/coqui_tts-January-21-2022_05+16AM-7f1a2378/config.json" \
    --out_path "/home/spow12/codes/TTS/coqui_TTS/output.wav"  \
    --speaker_idx 'ムラサメ' \
    --use_cuda "True"
    # --list_speaker_idxs "True"