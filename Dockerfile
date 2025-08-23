# ---------- 前端构建阶段 ----------
FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile
COPY frontend ./
RUN pnpm build

# ---------- 后端运行阶段 ----------
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 安装运行时库（仅运行态必须项；去掉编译工具与 *-dev）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    zlib1g \
    curl \
    # OpenCV 运行时所需（即便使用 headless 版，也常需 libGL）
    libgl1 \
    libxext6 \
    libsm6 \
    libxrender1 \
   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖清单并安装（充分利用缓存）
COPY requirements.txt /app/
# 优先安装 CPU 版 torch/torchvision，避免被其他依赖拉到 CUDA 版
# 如你的 requirements.txt 里包含 torch/torchvision，请移除或改为 +cpu 变体
ARG TORCH_VER=2.5.1
ARG TV_VER=0.20.1
RUN pip install "torch==${TORCH_VER}+cpu" "torchvision==${TV_VER}+cpu" --index-url https://download.pytorch.org/whl/cpu
# 安装其余依赖；opencv 使用 headless 变体，避免 GUI 依赖
RUN pip install -r requirements.txt && \
    pip install "opencv-python-headless>=4.8"

# 拷贝后端代码
COPY backend /app/backend

# 拷贝已构建的前端导出产物（确保与后端 [backend/app.py](backend/app.py) 的路径一致）
COPY --from=frontend-builder /app/frontend/out /app/frontend/out

# 模型不再 COPY 进入镜像，改由 docker-compose 卷挂载提供（减小镜像）
# 见 docker-compose.yml: volumes: ./model:/app/model:ro

EXPOSE 8000

# 使用 exec 形式启动 uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]