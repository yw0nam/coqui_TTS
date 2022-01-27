python ./TTS/bin/synthesize.py --text "あ___すみません。アタシはここいらに詳しくないんですよ" \
    --model_path "./experiment/visual_novel_vits-January-26-2022_06+18PM-a9003d8b/best_model.pth.tar" \
    --config_path "./experiment/visual_novel_vits-January-26-2022_06+18PM-a9003d8b/config.json" \
    --out_path "/home/spow12/codes/TTS/coqui_TTS/output.wav"  \
    # --vocoder_path "./experiment/visual_novel_melgan_256/coqui_tts-January-23-2022_10+16PM-74ae0045/best_model.pth.tar" \
    # --vocoder_config_path "./experiment/visual_novel_melgan_256/coqui_tts-January-23-2022_10+16PM-74ae0045/config.json" \
    # --use_cuda "True" \
    # --speaker_idx '芳乃' \
    # --speakers_file_path "./experiment/visual_novel_vits-January-25-2022_11+28PM-dbc9829f/speakers.json" \