import sys
from pathlib import Path

import torchaudio
from tqdm import tqdm

sys.path.append("CosyVoice")
sys.path.append("CosyVoice/third_party/Matcha-TTS")
sys.path.append("vllm")

Path("./output").mkdir(exist_ok=True)

from CosyVoice.cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
from vllm import ModelRegistry

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
from CosyVoice.cosyvoice.utils.common import set_all_random_seed
from CosyVoice.cosyvoice.utils.file_utils import load_wav

cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B",
    load_jit=True,
    load_trt=False,
    load_vllm=True,
    fp16=True,
)

prompt_speech_16k = load_wav("./CosyVoice/asset/zero_shot_prompt.wav", 16000)
for i in tqdm(range(10)):
    set_all_random_seed(i)
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            "希望你以后能够做的比我还好呦。",
            prompt_speech_16k,
            stream=False,
        )
    ):
        torchaudio.save(
            "output/vllm_zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )
