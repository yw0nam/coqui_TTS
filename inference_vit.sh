python ./TTS/bin/synthesize.py --text "yokatta-n-rerorero-chutchu-recharero" \
    --speakers_file_path "./experiment/coqui_tts-January-23-2022_04+09AM-74ae0045/speakers.json"\
    --model_path "experiment/coqui_tts-January-23-2022_04+09AM-74ae0045/best_model.pth.tar" \
    --config_path "experiment/coqui_tts-January-23-2022_04+09AM-74ae0045/config.json" \
    --out_path "/home/spow12/codes/TTS/coqui_TTS/output.wav"  \
    --speaker_idx '芳乃' \
    --use_cuda "True" \
    # --vocoder_path "./experiment/visual_novel_melgan_256/coqui_tts-January-23-2022_10+16PM-74ae0045/best_model_19680.pth.tar" \
    # --vocoder_config_path "./experiment/visual_novel_melgan_256/coqui_tts-January-23-2022_10+16PM-74ae0045/config.json"