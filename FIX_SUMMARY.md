# CosyVoice API 修复总结

## 修复时间
2026-01-13

## 问题描述

### 1. 非流式接口报错
**错误信息**: `soundfile.LibsndfileError: Error opening <_io.BytesIO ...>: Format not recognised.`

**原因**: CosyVoice 返回的音频张量维度是 `[channels, samples]` (通常是 `[1, N]`),而 `soundfile.write()` 期望的是一维数组 `[samples]`。

### 2. 流式输出问题
- 后端流式输出使用了 `.numpy().tobytes()`,没有先 flatten,导致音频数据格式不一致
- 前端流式播放功能未实现,只是简单回退到同步模式

## 修复方案

### 后端修复

#### 1. `app/main.py` - 非流式输出修复 (第95行)
```python
# 修复前
b64 = wav_to_base64(full_audio.numpy(), model.sample_rate)

# 修复后
b64 = wav_to_base64(full_audio.numpy().flatten(), model.sample_rate)
```

**说明**: 参考 `webui.py` 的做法,使用 `.flatten()` 将音频张量展平为一维数组。

#### 2. `app/main.py` - 流式输出修复 (第58行)
```python
# 修复前
audio_data = chunk['tts_speech'].numpy().tobytes()

# 修复后
audio_data = chunk['tts_speech'].numpy().flatten().tobytes()
```

**说明**: 确保流式输出的音频数据也是一维数组格式。

#### 3. `app/main.py` - Zero Shot 模式格式化 (第43行, 第75行)
```python
# 为 zero_shot 模式添加正确的 prompt 格式
formatted_tts_text = f"You are a helpful assistant.<|endofprompt|>{req.prompt_text}"
it = model.inference_zero_shot(req.text, formatted_tts_text, req.prompt_wav_path, ...)
```

**说明**: 参考 `example.py` 中 CosyVoice3 的使用方式,为 prompt_text 添加正确的格式前缀。

### 前端修复

#### 1. `static/app.js` - 实现真正的流式播放
**核心功能**:
- 使用 Web Audio API 实现实时音频播放
- 接收到第一个音频块后立即开始播放
- 播放完成后生成可下载的完整 WAV 文件

**关键代码**:
```javascript
const handleStreaming = async () => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const sampleRate = 22050;
    
    // 读取流式响应
    const reader = res.body.getReader();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // 转换为 Float32Array 并立即播放
        const float32Array = new Float32Array(value.buffer);
        playAudioChunk(float32Array);
        audioBuffers.push(float32Array);
    }
    
    // 生成完整的 WAV 文件供下载
    const wavBlob = createWavBlob(combinedArray, sampleRate);
    audioUrl.value = URL.createObjectURL(wavBlob);
};
```

**新增功能**:
- `playAudioChunk()`: 使用 Web Audio API 播放音频块
- `createWavBlob()`: 将 Float32Array 转换为 WAV 格式的 Blob
- 支持中止流式生成 (AbortController)

#### 2. `static/index.html` - 更新 UI 提示
```html
<!-- 区分流式和非流式的加载提示 -->
<p class="mt-4" v-if="form.stream">
    <i class="fas fa-broadcast-tower mr-2"></i>
    流式生成中,音频将实时播放...
</p>
<p class="mt-4" v-else>语音正在飞速合成中,请稍候...</p>
```

## 技术要点

### 音频数据格式
- **CosyVoice 输出**: `torch.Tensor` 形状为 `[1, N]` (channels, samples)
- **soundfile 期望**: 一维 numpy 数组 `[N]`
- **解决方案**: 使用 `.numpy().flatten()` 转换

### 流式播放原理
1. 后端返回原始 PCM 数据 (Float32 格式)
2. 前端使用 `fetch` API 的 `ReadableStream` 读取数据
3. 将字节流转换为 `Float32Array`
4. 使用 Web Audio API 的 `AudioBuffer` 和 `AudioBufferSource` 实时播放
5. 缓存所有音频块,最后合成完整的 WAV 文件

### WAV 文件格式
- 44 字节 WAV 头部
- PCM 格式,单声道,16位
- Float32 转 Int16: `s < 0 ? s * 0x8000 : s * 0x7FFF`

## 测试建议

### 非流式模式测试
```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好,我是通义实验室语音团队全新推出的生成式语音大模型",
    "mode": "sft",
    "speaker": "中文女",
    "stream": false
  }'
```

### 流式模式测试
1. 打开浏览器访问 `http://localhost:8000`
2. 勾选"流式推理"选项
3. 输入文本并点击"立即生成语音"
4. 观察音频是否实时播放

## 参考文件
- `webui.py`: 音频数据处理的正确方式 (`.numpy().flatten()`)
- `example.py`: CosyVoice3 的正确使用方式 (prompt 格式化)
- Web Audio API 文档: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API

## 注意事项
1. 所有修改仅限于 `app/` 和 `static/` 目录,未触及 `cosyvoice` 核心库
2. 流式播放需要浏览器支持 Web Audio API
3. 流式模式下音频会实时播放,同时在播放完成后提供完整文件下载
