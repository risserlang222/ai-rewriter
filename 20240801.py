# https://qiita.com/sayo0127/items/e22fdc229d2dfd879f75

# pyannoteはwavしか動作しないかもしれない。copilotによると、wavしか対応していない。
# TODO:変換所要時間を出せるとよい
# TODO:変換対象ファイルの有無を発見したい
# TODO:ffmpegで変換してモノラルwavを作るには？？？

# ライブラリの読み込み
import whisper
import os
import torch
from pyannote.audio import Pipeline
from pyannote.audio import Audio

from pydub import AudioSegment
from pydub.silence import split_on_silence

import tempfile

# whisperモデルの設定
model = whisper.load_model("medium")
 
# パイプラインの設定
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
# 明示的でないと、cudaを使用できないとノウハウあり。　https://qiita.com/daiki7010/items/c26c0818997229767d2f
pipeline.to(torch.device("cuda"))



def remove_silence(input_path, output_path):
    # 音声ファイルの読み込み
    sound = AudioSegment.from_file(input_path)

    # 元の音声の長さを計算し、分単位で表示
    org_ms = len(sound)
    print('original: {:.2f} [min]'.format(org_ms/60/1000))

    # 無音部分を抽出し、音声を分割
    chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-55, keep_silence=100)

    # 無音部分を除去した新しい音声を作成
    no_silence_audio = AudioSegment.empty()
    for chunk in chunks:
        no_silence_audio += chunk

    # 無音部分を除去した音声を出力
    no_silence_audio.export(output_path, format="mp3")
    org_ms = len(no_silence_audio)
    print('removed: {:.2f} [min]'.format(org_ms/60/1000))
 



# 入力音声ファイルのパス
source_file = "audio_mono.wav"

# 無音部分を除去した音声を保存するための一時ファイルを作成
no_silence_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
# 無音部分を除去
remove_silence(source_file, no_silence_audio_file.name)

diarization = pipeline(no_silence_audio_file.name ,num_speakers=2)



audio = Audio(sample_rate=16000, mono=True)

for segment, _, speaker in diarization.itertracks(yield_label=True):
    # 音声ファイルから話者のセグメントを切り出す
    waveform, sample_rate = audio.crop(no_silence_audio_file.name, segment)
    text = model.transcribe(waveform.squeeze().numpy())["text"]
    print(f"[{segment.start:03.1f}s - {segment.end:03.1f}s] {speaker}: {text}")
