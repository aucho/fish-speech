# Fish Speech API 文档

Fish Speech 提供了完整的 HTTP API 接口，支持文本转语音、参考音频管理、异步任务处理等功能。

## 目录

- [基础信息](#基础信息)
- [认证](#认证)
- [API 端点](#api-端点)
  - [健康检查](#健康检查)
  - [文本转语音 (TTS)](#文本转语音-tts)
  - [VQGAN 编码/解码](#vqgan-编码解码)
  - [参考音频管理](#参考音频管理)
  - [异步任务处理](#异步任务处理)
- [请求/响应格式](#请求响应格式)
- [客户端示例](#客户端示例)

## 基础信息

### 服务器启动

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --api-key YOUR_API_KEY  # 可选，设置后需要认证
```

### 服务器参数

- `--listen`: 监听地址和端口 (默认: `127.0.0.1:8080`)
- `--llama-checkpoint-path`: LLaMA 模型检查点路径
- `--decoder-checkpoint-path`: 解码器检查点路径
- `--decoder-config-name`: 解码器配置名称 (默认: `modded_dac_vq`)
- `--device`: 设备类型 (默认: `cuda`)
- `--half`: 使用半精度浮点数
- `--compile`: 启用 torch.compile 加速 (约提速10倍)
- `--max-text-length`: 最大文本长度限制 (0 表示无限制)
- `--workers`: 工作进程数 (默认: 1)
- `--api-key`: API 密钥 (可选，设置后需要 Bearer Token 认证)

### 内容类型

API 支持以下内容类型：

- `application/msgpack` (推荐，更高效)
- `application/json`
- `multipart/form-data` (用于文件上传)

可以通过以下方式指定响应格式：

- URL 参数: `?format=json` 或 `?format=msgpack`
- Accept 头: `Accept: application/json` 或 `Accept: application/msgpack`

## 认证

如果服务器设置了 `--api-key`，所有请求都需要在请求头中包含 Bearer Token：

```http
Authorization: Bearer YOUR_API_KEY
```

## API 端点

### 健康检查

#### GET/POST `/v1/health`

检查服务器健康状态。

**请求示例:**

```bash
curl http://127.0.0.1:8080/v1/health
```

**响应示例:**

```json
{
  "status": "ok"
}
```

---

### 文本转语音 (TTS)

#### POST `/v1/tts`

将文本转换为语音。

**请求体 (ServeTTSRequest):**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 要合成的文本 |
| `chunk_length` | int | 否 | 200 | 分块长度 (100-300) |
| `format` | string | 否 | "wav" | 音频格式: "wav", "pcm", "mp3" |
| `references` | array | 否 | [] | 参考音频列表 (见下方) |
| `reference_id` | string | 否 | null | 参考音频 ID (本地: 文件夹名称) |
| `seed` | int | 否 | null | 随机种子 (None 表示随机) |
| `use_memory_cache` | string | 否 | "off" | 内存缓存: "on", "off" |
| `normalize` | bool | 否 | true | 文本标准化 (提高数字稳定性) |
| `streaming` | bool | 否 | false | 是否启用流式响应 |
| `max_new_tokens` | int | 否 | 1024 | 最大生成 token 数 (0 表示无限制) |
| `top_p` | float | 否 | 0.8 | Top-p 采样 (0.1-1.0) |
| `repetition_penalty` | float | 否 | 1.1 | 重复惩罚 (0.9-2.0) |
| `temperature` | float | 否 | 0.8 | 采样温度 (0.1-1.0) |

**参考音频对象 (ServeReferenceAudio):**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | bytes | 是 | 音频文件字节 (支持 Base64 编码字符串) |
| `text` | string | 是 | 参考文本 |

**请求示例 (使用 msgpack):**

```python
import requests
import ormsgpack
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

data = ServeTTSRequest(
    text="你好，这是测试文本",
    format="wav",
    reference_id="my_voice",  # 使用已保存的参考音频
    streaming=False,
)

response = requests.post(
    "http://127.0.0.1:8080/v1/tts",
    params={"format": "msgpack"},
    data=ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
    headers={
        "authorization": "Bearer YOUR_API_KEY",
        "content-type": "application/msgpack",
    },
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
```

**请求示例 (使用 JSON):**

```python
import requests

data = {
    "text": "你好，这是测试文本",
    "format": "wav",
    "reference_id": "my_voice",
    "streaming": False,
}

response = requests.post(
    "http://127.0.0.1:8080/v1/tts?format=json",
    json=data,
    headers={"authorization": "Bearer YOUR_API_KEY"},
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
```

**请求示例 (使用参考音频):**

```python
from fish_speech.utils.file import audio_to_bytes, read_ref_text

# 使用本地参考音频文件
data = ServeTTSRequest(
    text="你好，这是测试文本",
    format="wav",
    references=[
        ServeReferenceAudio(
            audio=audio_to_bytes("ref_audio.wav"),
            text=read_ref_text("ref_text.txt")
        )
    ],
)

response = requests.post(
    "http://127.0.0.1:8080/v1/tts",
    params={"format": "msgpack"},
    data=ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
    headers={
        "authorization": "Bearer YOUR_API_KEY",
        "content-type": "application/msgpack",
    },
)
```

**流式响应示例:**

```python
import wave
import pyaudio

data = ServeTTSRequest(
    text="这是一段较长的文本，适合流式输出...",
    format="wav",
    streaming=True,
)

response = requests.post(
    "http://127.0.0.1:8080/v1/tts",
    params={"format": "msgpack"},
    data=ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
    headers={
        "authorization": "Bearer YOUR_API_KEY",
        "content-type": "application/msgpack",
    },
    stream=True,
)

# 实时播放音频
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=44100,
    output=True,
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        stream.write(chunk)

stream.stop_stream()
stream.close()
p.terminate()
```

**响应:**

- 成功 (200): 返回音频文件 (二进制数据)
- Content-Type: `audio/wav`, `audio/mpeg` (mp3), 或 `application/octet-stream`
- Content-Disposition: `attachment; filename=audio.{format}`

**注意事项:**

- 流式响应仅支持 WAV 格式
- `reference_id` 优先级高于 `references` 数组
- 如果设置了 `max_text_length`，文本长度超过限制将返回 400 错误

---

### VQGAN 编码/解码

#### POST `/v1/vqgan/encode`

将音频编码为 VQGAN tokens。

**请求体 (ServeVQGANEncodeRequest):**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audios` | array[bytes] | 是 | 音频文件字节数组 (支持 wav, mp3 等格式) |

**请求示例:**

```python
import requests
import ormsgpack
from fish_speech.utils.schema import ServeVQGANEncodeRequest
from fish_speech.utils.file import audio_to_bytes

data = ServeVQGANEncodeRequest(
    audios=[audio_to_bytes("audio.wav")]
)

response = requests.post(
    "http://127.0.0.1:8080/v1/vqgan/encode",
    params={"format": "msgpack"},
    data=ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
    headers={
        "authorization": "Bearer YOUR_API_KEY",
        "content-type": "application/msgpack",
    },
)

if response.status_code == 200:
    result = ormsgpack.unpackb(response.content)
    tokens = result["tokens"]  # list[list[list[int]]]
```

**响应 (ServeVQGANEncodeResponse):**

```json
{
  "tokens": [[[1, 2, 3, ...], ...], ...]
}
```

#### POST `/v1/vqgan/decode`

将 VQGAN tokens 解码为音频。

**请求体 (ServeVQGANDecodeRequest):**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tokens` | array[array[array[int]]] | 是 | VQGAN token 数组 |

**请求示例:**

```python
import requests
import ormsgpack
from fish_speech.utils.schema import ServeVQGANDecodeRequest

data = ServeVQGANDecodeRequest(
    tokens=[[[1, 2, 3, ...], ...], ...]
)

response = requests.post(
    "http://127.0.0.1:8080/v1/vqgan/decode",
    params={"format": "msgpack"},
    data=ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
    headers={
        "authorization": "Bearer YOUR_API_KEY",
        "content-type": "application/msgpack",
    },
)

if response.status_code == 200:
    result = ormsgpack.unpackb(response.content)
    audios = result["audios"]  # list[bytes] (PCM float16 格式)
```

**响应 (ServeVQGANDecodeResponse):**

```json
{
  "audios": ["<bytes>", ...]
}
```

---

### 参考音频管理

#### POST `/v1/references/add`

添加新的参考音频。

**请求体 (multipart/form-data):**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 参考音频 ID (1-255 字符，仅字母数字、连字符、下划线、空格) |
| `audio` | file | 是 | 音频文件 |
| `text` | string | 是 | 参考文本 |

**请求示例:**

```python
import requests

with open("ref_audio.wav", "rb") as audio_file:
    files = {"audio": audio_file}
    data = {
        "id": "my_voice",
        "text": "这是参考文本"
    }
    
    response = requests.post(
        "http://127.0.0.1:8080/v1/references/add",
        files=files,
        data=data,
        headers={"authorization": "Bearer YOUR_API_KEY"},
    )
    
    print(response.json())
```

**响应 (AddReferenceResponse):**

```json
{
  "success": true,
  "message": "Reference voice 'my_voice' added successfully",
  "reference_id": "my_voice"
}
```

**错误响应:**

- 400: 输入验证失败
- 409: 参考 ID 已存在
- 500: 服务器错误

#### GET `/v1/references/list`

获取所有可用的参考音频 ID 列表。

**请求示例:**

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://127.0.0.1:8080/v1/references/list
```

**响应 (ListReferencesResponse):**

```json
{
  "success": true,
  "reference_ids": ["voice1", "voice2", "voice3"],
  "message": "Found 3 reference voices"
}
```

#### DELETE `/v1/references/delete`

删除指定的参考音频。

**请求体:**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `reference_id` | string | 是 | 要删除的参考音频 ID |

**请求示例:**

```python
import requests

response = requests.delete(
    "http://127.0.0.1:8080/v1/references/delete",
    json={"reference_id": "my_voice"},
    headers={"authorization": "Bearer YOUR_API_KEY"},
)

print(response.json())
```

**响应 (DeleteReferenceResponse):**

```json
{
  "success": true,
  "message": "Reference voice 'my_voice' deleted successfully",
  "reference_id": "my_voice"
}
```

**错误响应:**

- 400: 输入验证失败
- 404: 参考 ID 不存在
- 500: 服务器错误

#### POST `/v1/references/update`

重命名参考音频。

**请求体:**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `old_reference_id` | string | 是 | 旧的参考音频 ID |
| `new_reference_id` | string | 是 | 新的参考音频 ID |

**请求示例:**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8080/v1/references/update",
    json={
        "old_reference_id": "old_name",
        "new_reference_id": "new_name"
    },
    headers={"authorization": "Bearer YOUR_API_KEY"},
)

print(response.json())
```

**响应 (UpdateReferenceResponse):**

```json
{
  "success": true,
  "message": "Reference voice renamed from 'old_name' to 'new_name' successfully",
  "old_reference_id": "old_name",
  "new_reference_id": "new_name"
}
```

**错误响应:**

- 400: 输入验证失败
- 404: 旧参考 ID 不存在
- 409: 新参考 ID 已存在
- 500: 服务器错误

---

### 异步任务处理

#### POST `/generate_audio_enhanced_async`

创建异步音频生成任务。

**请求体 (AsyncGenerateRequest):**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `step_id` | string | 是 | - | 任务唯一标识符 |
| `text` | string | 是 | - | 要合成的文本 |
| `chunk_length` | int | 否 | 200 | 分块长度 (100-300) |
| `format` | string | 否 | "mp3" | 音频格式: "wav", "pcm", "mp3" |
| `references` | array | 否 | [] | 参考音频列表 |
| `reference_id` | string | 否 | null | 参考音频 ID |
| `seed` | int | 否 | null | 随机种子 |
| `use_memory_cache` | string | 否 | "off" | 内存缓存: "on", "off" |
| `normalize` | bool | 否 | true | 文本标准化 |
| `max_new_tokens` | int | 否 | 2048 | 最大生成 token 数 |
| `top_p` | float | 否 | 0.8 | Top-p 采样 (0.1-1.0) |
| `repetition_penalty` | float | 否 | 1.1 | 重复惩罚 (0.9-2.0) |
| `temperature` | float | 否 | 0.8 | 采样温度 (0.1-1.0) |

**请求示例:**

```python
import requests
import ormsgpack
from fish_speech.utils.schema import AsyncGenerateRequest

data = AsyncGenerateRequest(
    step_id="task_001",
    text="这是一段需要异步处理的文本",
    format="mp3",
    reference_id="my_voice",
)

response = requests.post(
    "http://127.0.0.1:8080/generate_audio_enhanced_async",
    params={"format": "msgpack"},
    data=ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
    headers={
        "authorization": "Bearer YOUR_API_KEY",
        "content-type": "application/msgpack",
    },
)

if response.status_code == 200:
    result = ormsgpack.unpackb(response.content)
    print(f"Task created: {result['step_id']}, Status: {result['status']}")
```

**响应 (AsyncGenerateResponse):**

```json
{
  "success": true,
  "message": "Task created and started",
  "step_id": "task_001",
  "status": "pending"  // pending, processing, completed, failed, cancelled
}
```

#### GET `/get_task_status/{step_id}`

查询任务状态。

**请求示例:**

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://127.0.0.1:8080/get_task_status/task_001
```

**响应 (TaskStatusResponse):**

```json
{
  "success": true,
  "step_id": "task_001",
  "status": "completed",  // pending, processing, completed, failed, cancelled
  "created_at": 1234567890.123,
  "started_at": 1234567891.456,
  "completed_at": 1234567900.789,
  "download_url": "/download_result/task_001",
  "error": null
}
```

**任务状态说明:**

- `pending`: 任务已创建，等待执行
- `processing`: 任务正在处理中
- `completed`: 任务已完成
- `failed`: 任务执行失败
- `cancelled`: 任务已取消
- `not_found`: 任务不存在 (404)

#### GET `/download_result/{step_id}`

下载任务生成的结果文件。

**请求示例:**

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://127.0.0.1:8080/download_result/task_001 \
     -o output.mp3
```

**响应:**

- 成功 (200): 返回音频文件 (二进制数据)
- 404: 任务不存在
- 400: 任务未完成

#### POST `/stop_generation`

停止所有正在运行的生成任务。

**请求示例:**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8080/stop_generation",
    headers={"authorization": "Bearer YOUR_API_KEY"},
)

print(response.json())
```

**响应 (StopTaskResponse):**

```json
{
  "success": true,
  "message": "Stopped 3 task(s)"
}
```

#### POST `/stop_async_task/{step_id}`

停止指定的异步任务。

**请求示例:**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8080/stop_async_task/task_001",
    headers={"authorization": "Bearer YOUR_API_KEY"},
)

print(response.json())
```

**响应 (StopTaskResponse):**

```json
{
  "success": true,
  "message": "Task task_001 stopped",
  "step_id": "task_001"
}
```

**错误响应:**

- 404: 任务不存在

---

## 请求/响应格式

### 数据序列化

API 支持两种数据格式：

1. **MessagePack** (推荐)
   - 更高效，体积更小
   - Content-Type: `application/msgpack`
   - 使用 `ormsgpack` 库进行序列化/反序列化

2. **JSON**
   - 更易读，兼容性更好
   - Content-Type: `application/json`
   - 标准 JSON 格式

### 错误响应格式

所有错误响应都遵循统一格式：

```json
{
  "detail": "错误描述信息"
}
```

常见 HTTP 状态码：

- `200`: 成功
- `400`: 请求参数错误
- `401`: 认证失败
- `404`: 资源不存在
- `409`: 资源冲突 (如 ID 已存在)
- `500`: 服务器内部错误

---

## 客户端示例

### Python 客户端

使用提供的 `api_client.py` 工具：

```bash
python tools/api_client.py \
    --url http://127.0.0.1:8080/v1/tts \
    --text "你好，这是测试文本" \
    --reference_id my_voice \
    --format wav \
    --output output \
    --api-key YOUR_API_KEY
```

### cURL 示例

```bash
# 健康检查
curl http://127.0.0.1:8080/v1/health

# TTS (需要将请求体序列化为 msgpack 或 json)
curl -X POST http://127.0.0.1:8080/v1/tts?format=json \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -d '{"text": "测试文本", "format": "wav"}'

# 列出参考音频
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://127.0.0.1:8080/v1/references/list
```

### JavaScript/TypeScript 示例

```javascript
// 使用 fetch API
async function generateSpeech(text, referenceId) {
  const response = await fetch('http://127.0.0.1:8080/v1/tts?format=json', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
      text: text,
      format: 'wav',
      reference_id: referenceId
    })
  });
  
  if (response.ok) {
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
  }
}
```

---

## 最佳实践

1. **使用 MessagePack 格式**: 对于生产环境，推荐使用 MessagePack 以获得更好的性能
2. **设置 API 密钥**: 在生产环境中务必设置 API 密钥以保护服务
3. **错误处理**: 始终检查响应状态码并处理错误情况
4. **流式响应**: 对于长文本，使用流式响应可以更快地获得结果
5. **参考音频缓存**: 对于频繁使用的参考音频，设置 `use_memory_cache="on"` 可以提高性能
6. **异步任务**: 对于长时间运行的生成任务，使用异步接口可以避免超时

---

## 常见问题

**Q: 如何设置参考音频？**

A: 有两种方式：
1. 使用 API 上传: `POST /v1/references/add`
2. 手动创建: 在服务器 `references` 目录下创建文件夹，放入音频文件和文本文件

**Q: 流式响应和普通响应有什么区别？**

A: 流式响应会实时返回音频数据块，可以边生成边播放，适合长文本。普通响应会等待全部生成完成后一次性返回。

**Q: 如何提高生成速度？**

A: 
- 使用 `--compile` 参数启动服务器 (约提速10倍)
- 使用 GPU 加速
- 对于重复使用的参考音频，启用内存缓存

**Q: 支持哪些音频格式？**

A: 输入支持 wav, mp3 等常见格式。输出支持 wav, mp3, pcm 格式。

---

## 更新日志

- **v1.5.0**: 当前版本
  - 支持异步任务处理
  - 支持参考音频管理 API
  - 支持流式响应
  - 支持多种音频格式

