import sys
from pathlib import Path

import torchaudio

ROOT_DIR = Path.cwd()
sys.path.append(f"{ROOT_DIR}/")
sys.path.append(f"{ROOT_DIR}/third_party/Matcha-TTS")
sys.path.append(f"{ROOT_DIR}/third_party/async_cosyvoice")
sys.path.append(f"{ROOT_DIR}/third_party/vllm")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# prompt_speech_16k = load_wav("./asset/zero_shot_prompt.wav", 16000)
prompt_speech_16k = load_wav("./asset/cross_lingual_prompt.wav", 16000)
prompt_txt = [
    "收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
]

cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B",
    load_jit=True,
    load_trt=False,
    load_vllm=False,
    fp16=True,
)

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for _ in range(4):
    for i, j in enumerate(
        cosyvoice.inference_cross_lingual(prompt_txt[0], prompt_speech_16k, stream=False)
    ):
        Path("./output").mkdir(exist_ok=True)
        torchaudio.save(
            f"output/fine_grained_control_{i}.wav",
            j["tts_speech"],
            cosyvoice.sample_rate,
        )
