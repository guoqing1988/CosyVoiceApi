# CosyVoice 高并发语音生成 API 服务

基于 FunAudioLLM CosyVoice3 + vLLM + FastAPI 构建的工业级语音合成服务，支持：

- 高并发推理（vLLM CUDA Graph + KV Cache）
- HTTP 流式音频输出
- WebSocket 实时语音流
- 商用部署（多卡 / 多进程 / 负载均衡）
- 低延迟首帧返回（<300ms）

项目地址：
https://github.com/guoqing1988/CosyVoiceApi

---

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
├── static/              # 前端静态资源
│   ├── index.html       # UI 界面
│   ├── app.js           # 界面逻辑
├── cosyvoice/           # 原始 CosyVoice 引擎
├── vllm_example.py      # vllm 运行测试
├── requirements.txt
├── APP.md               # 项目描述文档
└── README.md            # 原始 https://github.com/FunAudioLLM/CosyVoice 项目介绍

```

---

## 3. API 接口设计

### 3.1 基础接口

#### [GET] 健康检查
- **URL**: `/v1/health`
- **描述**: 返回服务运行状态、GPU 状态及加载的模型类名。
- **返回示例**:
```json
{
  "status": "ok",
  "gpu": true,
  "model": "CosyVoice3"
}
```

#### [GET] 获取音色列表
- **URL**: `/v1/speakers`
- **描述**: 返回当前模型支持的所有预训练音色。
- **返回示例**:
```json
{
  "speakers": ["中文女", "中文男", "日语男", ...]
}
```

---

### 3.2 语音生成接口 (TTS)

#### [POST] 统一合成接口
- **URL**: `/v1/tts`
- **Content-Type**: `application/json`
- **描述**: 支持多种模式的同步或流式语音合成。

**请求参数详解:**

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `text` | string | - | **必填**。需要合成的目标文本。 |
| `mode` | string | `sft` | 合成模式。可选值：<br> - `sft`: 预训练音色<br> - `zero_shot`: 3s 极速复刻<br> - `cross_lingual`: 跨语种复刻<br> - `instruct`: 自然语言控制<br> - `vc`: 声音转换 |
| `speaker` | string | `中文女` | `sft` 或 `instruct` 模式下的音色名称。 |
| `prompt_text` | string | `""` | `zero_shot` 模式下 prompt 音频对应的文本内容。 |
| `prompt_wav_path` | string | `""` | Prompt 音频文件的服务器本地绝对路径。用于复刻或控制音色。 |
| `instruct_text` | string | `""` | `instruct` 模式下的情感/风格控制文本（如“请用开心的语气说”）。 |
| `source_wav_path` | string | `""` | `vc` (声音转换) 模式下的源音频服务器本地绝对路径。 |
| `stream` | boolean | `false` | 是否开启流式返回。 |
| `speed` | float | `1.0` | 合成语速，范围 0.5 - 2.0。 |

**响应说明:**
1. **非流式 (`stream: false`)**:
   - 返回 `JSON` 格式，包含 Base64 编码的音频数据。
   - 示例: `{"audio": "UklGRi...", "sample_rate": 22050}`
2. **流式 (`stream: true`)**:
   - 返回 `audio/pcm` 类型的二进制流分片。
   - 客户端应按顺序接收分片并实时播放。

---

#### [POST] 流式合成专用接口
- **URL**: `/v1/tts/stream`
- **描述**: 功能与 `/v1/tts` 一致，但强制将 `stream` 设为 `true`。
- **返回**: `audio/pcm` 二进制流。

---

### 3.3 实时交互接口

#### [WebSocket] 实时语音合成
- **URL**: `ws://[host]:[port]/ws/v1/tts`
- **交互流程**:
  1. 客户端建立 WebSocket 连接。
  2. 客户端发送 JSON 格式的配置（参数同 `/v1/tts`）。
  3. 服务端连续推送音频二进制数据分片 (`Binary Frame`)。
  4. 生成结束后，服务端发送一个 JSON 消息：`{"done": true}`。

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
export COSYVOICE_MODEL_DIR=/data/www/ComfyUI/models/cosyvoice/Fun-CosyVoice3-0.5B/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
/data/www/wwwroot/CosyVoiceApi/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

生产建议：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools
```

---

## 6. UI 控制台

项目内置了一个功能完备的 Web 控制台，方便测试各种模式。
- **访问地址**: `http://localhost:8000/index.html`
- **技术栈**: Vue 3 + Tailwind CSS。
- **功能**: 支持所有推理模式切换、路径输入、语速调节、在线试听及音频下载。

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
