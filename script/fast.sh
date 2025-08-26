git submodule add -b releases/v0.9.0 https://github.com/vllm-project/vllm.git third_party/vllm
git submodule add https://github.com/qi-hua/async_cosyvoice.git third_party/async_cosyvoice



git clone https://gitee.com/ascend/ModelZoo-PyTorch.git && cd ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/CosyVoice2
git clone https://github.com/FunAudioLLM/CosyVoice && cd CosyVoice && git reset --hard fd45708 && git submodule update --init --recursive
platform=800I
git apply ../${platform}/diff_CosyVoice_${platform}.patch
# 将infer.py复制到CosyVoice中
cp ../infer.py ./
git clone -b v4.37.0 https://github.com/huggingface/transformers.git
mv ../${platform}/modeling_qwen2.py ./transformers/src/transformers/models/qwen2

pip3 install -r ../requirements.txt
apt-get install sox # centos版本 yum install sox

(
    # 下载安装包并解压
    wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz && \
        tar -zxvf openfst-1.8.3.tar.gz && cd openfst-1.8.3 && \
        ./configure --enable-far --enable-mpdt --enable-pdt && make -j$(nproc) && make install && \
        ls /usr/local/lib/libfstmpdtscript.so.26 && \
        export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH && \
        sudo ldconfig
)

# 安装WeTextProcessing
pip3 install WeTextProcessing==1.0.4.1