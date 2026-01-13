# 安装

```shell
uv venv --python 3.11
source .venv/bin/activate

uv pip install protobuf==4.25.8   
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install pydantic-settings

uv pip install \
  hyperpyyaml==1.1.0 \
  ruamel.yaml==0.17.21 \
  ruamel.yaml.clib==0.2.7 \
  pyyaml==6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


uv pip install matcha-tts -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 运行

```shell
export COSYVOICE_MODEL_DIR=/data/www/ComfyUI/models/cosyvoice/Fun-CosyVoice3-0.5B/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
/data/www/wwwroot/CosyVoiceApi/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

/data/www/wwwroot/CosyVoiceApi/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 
```
