# 安装
uv venv --python 3.11
source .venv/bin/activate

uv pip install protobuf==4.25.8   
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install pydantic-settings
