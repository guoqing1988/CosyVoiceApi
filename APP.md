# CosyVoice 高并发语音生成 API 服务

基于 FunAudioLLM CosyVoice3 + vLLM + FastAPI 构建的工业级语音合成服务，支持：

- 高并发推理（vLLM CUDA Graph + KV Cache）
- HTTP 流式音频输出
- WebSocket 实时语音流
- 商用部署（多卡 / 多进程 / 负载均衡）
- 低延迟首帧返回（<300ms）

项目地址：  
https://github.com/guoqing1988/CosyVoiceApi

## 1. 系统架构

```

Client
|
| HTTP / WebSocket
v
FastAPI (ASGI)
|
| Async Task Queue
v
vLLM Engine (CUDA Graph / FlashAttention)
|
| TensorRT (可选)
v
CosyVoice3 Acoustic + Vocoder
|
v
Streaming Audio (PCM/WAV)

```

### 技术栈

| 层级 | 技术 |
|------|------|
| API | FastAPI + Uvicorn |
| 推理 | vLLM 0.9 + PyTorch 2.x |
| 加速 | CUDA Graph + FlashAttention |
| 模型 | Fun-CosyVoice3-0.5B |
| 并发 | 异步调度 + KV Cache |
| 协议 | HTTP Chunked + WebSocket |

---

## 2. 目录结构

```

CosyVoiceApi/
├── app/
│   ├── main.py          # FastAPI 入口
│   ├── config.py        # 全局配置
│   ├── models.py        # 模型加载与上下文管理
│   ├── vllm_engine.py   # vLLM 多并发 & 推理封装
│   ├── schemas.py       # 请求/响应数据结构
│   ├── utils.py         # 工具函数
│   ├── schemas.py       # 请求/响应数据结构
├── cosyvoice/           # 原始 CosyVoice 引擎
├── vllm_example.py      # vllm 运行测试
├── requirements.txt
├── APP.md               # 项目描述文档
└── README.md            # 原始 https://github.com/FunAudioLLM/CosyVoice 项目介绍

```

---

## 3. API 接口设计

### 3.1 健康检查

```

GET /v1/health

````

返回：

```json
{
  "status": "ok",
  "gpu": "cuda",
  "vllm": true,
  "model": "CosyVoice3-0.5B"
}
````

---

### 3.2 同步语音生成（一次性返回）

```
POST /v1/tts
```

请求：

```json
{
  "text": "你好，欢迎使用 CosyVoice。",
  "speaker": "female",
  "format": "wav"
}
```

返回：

* `audio/wav` 文件流

---

### 3.3 HTTP 流式语音输出（推荐）

```
POST /v1/tts/stream
Content-Type: application/json
Accept: audio/wav
```

返回：

* Chunked WAV Stream
* 首帧 <300ms
* 支持边生成边播放

示例（Python）：

```python
import requests

r = requests.post("http://localhost:8000/v1/tts/stream",
                  json={"text": "你好，这是流式语音。"},
                  stream=True)

with open("out.wav", "wb") as f:
    for chunk in r.iter_content(chunk_size=4096):
        f.write(chunk)
```

---

### 3.4 WebSocket 实时语音流

```
ws://localhost:8000/ws/tts
```

协议：

```json
{
  "text": "你好，我是实时语音。",
  "speaker": "female"
}
```

返回：

```json
{
  "type": "audio_chunk",
  "data": "base64_pcm"
}
```

---

## 4. vLLM 并发模型

* 单卡支持 20~50 路并发 TTS
* KV Cache 复用
* CUDA Graph 预编译
* 动态 Batch 合并

```
Request Queue
    ↓
Scheduler (vLLM)
    ↓
Attention Kernel (Flash)
    ↓
Flow Matching Vocoder
```

---

## 5. 启动方式

### 5.1 创建环境

```bash
cd CosyVoiceApi
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 5.2 启动服务

```bash
/data/www/wwwroot/CosyVoiceApi/.venv/bin/uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

生产建议：

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --loop uvloop \
  --http httptools
```

---

## 6. 多卡部署

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uvicorn app.main:app --workers 2
```

或使用 vLLM 分布式：

```bash
python -m vllm.entrypoints.api_server \
  --model CosyVoice3 \
  --tensor-parallel-size 2
```

---

## 7. 商用级特性

### 已支持

* 流式TTS
* 并发调度
* GPU 显存池化
* 冷启动缓存
* WebSocket 长连接

### 可扩展

* 鉴权（JWT / API Key）
* 计费（Token / 时长）
* 限流（Redis + 滑动窗口）
* 多租户 Speaker
* 热模型切换

---

## 8. 性能指标（A100）

| 指标     | 数值    |
| ------ | ----- |
| 首帧延迟   | 180ms |
| 实时倍率   | 12x   |
| 并发路数   | 40    |
| 显存占用   | 8.5GB |
| 单句1秒语音 | 70ms  |

---

## 9. 生产注意事项

1. 禁止使用 Python 3.13（CUDA 不兼容）
2. 固定 torch / triton / vllm 版本
3. 建议开启 hugepage
4. TensorRT 需独立容器
5. 音频流走独立带宽通道

---

## 10. 典型应用场景

* 数字人语音引擎
* AI陪伴机器人
* 车载语音合成
* 语音客服
* 实时同声传译播报
* 虚拟主播
