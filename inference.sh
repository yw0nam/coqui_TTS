python ./TTS/bin/synthesize.py --text "a:sumimaseN.atashiwakokoiranikuwashikunaiNdesuyo" \
    --speakers_file_path "experiment/coqui_tts-January-25-2022_02+31AM-74ae0045/speakers.json"\
    --model_path "experiment/coqui_tts-January-25-2022_02+31AM-74ae0045/best_model.pth.tar" \
    --config_path "experiment/coqui_tts-January-25-2022_02+31AM-74ae0045/config.json" \
    --out_path "/home/spow12/codes/TTS/coqui_TTS/output.wav"  \
    --speaker_idx '芳乃' \
    # --vocoder_path "./experiment/visual_novel_melgan_256/coqui_tts-January-23-2022_10+16PM-74ae0045/best_model.pth.tar" \
    # --vocoder_config_path "./experiment/visual_novel_melgan_256/coqui_tts-January-23-2022_10+16PM-74ae0045/config.json"
    # --use_cuda "True" \