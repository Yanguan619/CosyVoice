import sys
from pathlib import Path

import torch
import torchaudio
from torch_npu.contrib import transfer_to_npu

sys.path.append("third_party/*")

Path("./output").mkdir(exist_ok=True)


from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B",
    load_jit=True,
    load_trt=True,
    load_vllm=False,
    fp16=True,
)
if True:
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.set_compile_mode(jit_compile=False)
    # ERROR
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    # 对hift模型结构进行torchair图模式适配
    cosyvoice.model.hift.remove_weight_norm()
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.hift.decode = torch.compile(
        cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend
    )

prompt_speech_16k = load_wav("./CosyVoice/asset/zero_shot_prompt.wav", 16000)
prompt_txt = [
    "收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    "全球每年有超过一百三十五万人，因吸烟而死亡",
]
# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for _ in range(10):
    for i, j in enumerate(
        cosyvoice.inference_cross_lingual(
            prompt_txt[0],
            prompt_speech_16k,
            stream=False,
        )
    ):
        torchaudio.save(
            "output/fine_grained_control_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )
