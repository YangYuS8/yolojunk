# ---------- 前端构建阶段 ----------
FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

# 启用 corepack 以使用 pnpm（如果你项目用 npm 可改为 npm ci）
RUN corepack enable

# 先复制锁文件与包清单，充分利用缓存
COPY frontend/package.json frontend/pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile

# 复制其余前端代码并构建（Next 15 + output: 'export' 会产出 frontend/out）
COPY frontend ./
RUN pnpm build

# ---------- 后端运行阶段 ----------
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 安装运行时库：OpenCV/ Pillow/ PyTorch 常见依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    libgomp1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖清单并安装（利用 Docker 层缓存）
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝后端代码
COPY backend /app/backend

# 拷贝已构建的前端导出产物（确保与后端 [backend/app.py](backend/app.py) 的路径一致）
COPY --from=frontend-builder /app/frontend/out /app/frontend/out

# 模型目录（可在 compose 中用卷挂载覆盖）
COPY model /app/model

EXPOSE 8000

# 使用 exec 形式启动 uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]