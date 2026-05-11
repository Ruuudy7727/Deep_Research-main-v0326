# FunASR 语音输入（科宝 Cobot / `server_plus.py`）

在输入框旁使用「语音」按钮：录音结束后上传到本机 FunASR 推理，**识别结果写入输入框**，可手动改字后再点「开始分析」。不替代主 LLM。

## 1. 环境（thinkdepth）

```bash
conda activate thinkdepth
cd /path/to/Deep_Research-main-v0326
python -m pip install -U "funasr>=1.0.0" "modelscope>=1.9.0"
```

可选 PyPI 镜像：

```bash
python -m pip install -U funasr modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```bash
python -c "import funasr; print('funasr', funasr.__version__)"
```

## 2. 系统依赖

浏览器上传一般为 **webm/opus**，服务端用 **ffmpeg** 转为 16k 单声道 WAV 再送模型：

```bash
ffmpeg -version
```

无 GPU 时请将 `FUNASR_DEVICE` 设为 `cpu`。

## 3. 环境变量（`.env` 可选）

| 变量 | 默认 | 说明 |
|------|------|------|
| `FUNASR_ENABLED` | `1` | 设为 `0` 关闭 ASR 接口与前端按钮 |
| `FUNASR_MODEL` | `paraformer-zh` | FunASR `AutoModel` 模型名或 ModelScope id |
| `FUNASR_DEVICE` | `cuda:0` | 无显卡改为 `cpu` |
| `FUNASR_HUB` | `ms` | 模型仓库，`ms` 或 `hf` |
| `FUNASR_MAX_UPLOAD_MB` | `32` | 单次上传上限 |
| `FUNASR_FFMPEG_TIMEOUT_SEC` | `120` | ffmpeg 转码超时（秒） |

## 4. 启动与接口

```bash
python server_plus.py
```

- `GET /api/asr/status`：是否启用、是否检测到 ffmpeg、模型是否已加载。
- `POST /api/asr/transcribe`：`multipart/form-data` 字段名 `audio`（文件）。

首次识别会下载模型，可能较慢，属正常现象。

## 5. 与 `requirements.txt` / `environment.yml`

已在仓库中增加 `funasr`、`modelscope` 条目；新建或更新 conda 环境时可随 `environment.yml` 一并安装。
