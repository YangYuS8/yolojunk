# jundek

一个基于 FastAPI + Ultralytics YOLO 的可回收物检测服务，前端使用 Next.js 15（Turbopack）+ Tailwind CSS v4 并以静态方式导出，由后端统一托管。

- 后端入口：[backend/app.py](backend/app.py)
- 前端配置：[frontend/next.config.ts](frontend/next.config.ts)
- 前端页面与组件：[frontend/app/page.tsx](frontend/app/page.tsx)、[frontend/components/UploadCard.tsx](frontend/components/UploadCard.tsx)
- 依赖清单：[requirements.txt](requirements.txt)
- 容器构建：多阶段 [Dockerfile](Dockerfile)，编排 [docker-compose.yml](docker-compose.yml)

## 已实现功能

- 图片上传与自动缩放显示（根据容器宽度/屏幕高度适配）
- YOLO 推理与检测框叠加渲染
- 可回收判断（基于类名关键词匹配）
- 前端静态导出（`output: 'export'`）由后端托管 `/_next` 与首页
- 健康检查与简易运维路由（favicon、Chrome DevTools 探测）

## 目录结构

- 后端 API 与模型加载：[`backend/app.py`](backend/app.py)
  - 读取环境变量 [`MODEL_PATH`](backend/app.py) 和 [`RECYCLABLE_CLASSES`](backend/app.py)
  - 路由：`GET /`（首页）、`POST /predict`（推理）、`GET /_next/*`（前端静态资源）
- 前端应用（静态导出）：[`frontend`](frontend/)
  - 配置：[`next.config.ts`](frontend/next.config.ts)（`output: 'export'`）
  - 全局样式（Tailwind v4）：[`app/globals.css`](frontend/app/globals.css)（`@import "tailwindcss";`）
  - PostCSS：[`postcss.config.mjs`](frontend/postcss.config.mjs)
- 容器与编排：[`Dockerfile`](Dockerfile)、[`docker-compose.yml`](docker-compose.yml)
- 依赖：[`requirements.txt`](requirements.txt)、`frontend/pnpm-lock.yaml`

## 本地开发

前置要求：

- Linux，Python 3.12
- Node.js 20+，pnpm（`corepack enable`）

步骤（推荐通过后端托管前端静态导出以避免 CORS）：

1) 安装 Python 依赖并准备模型
   ```bash
   cd /home/yangyus8/code/jundek
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   # 将模型权重放到 model/best.pt（或设置 MODEL_PATH）
   ```
2) 构建前端静态资源
   ```bash
   cd frontend
   pnpm install --frozen-lockfile
   pnpm build   # 生成 frontend/out
   ```
3) 启动后端
   ```bash
   cd ..
   uvicorn backend.app:app --reload --port 8000
   ```
4) 打开浏览器访问 http://127.0.0.1:8000/

说明：

- Tailwind v4 使用 `@import "tailwindcss";`（见 [`app/globals.css`](frontend/app/globals.css)），PostCSS 插件为 `@tailwindcss/postcss`（见 [`postcss.config.mjs`](frontend/postcss.config.mjs)）。
- 若需前端独立开发：`pnpm dev`（端口 3000），请将前端请求的 API 地址改为 `http://127.0.0.1:8000`（或为后端开启 CORS）。

## API

- POST `/predict`：接收表单文件字段 `file`（image/*）
  - 请求（curl 例）：
    ```bash
    curl -F "file=@/path/to/image.jpg" http://127.0.0.1:8000/predict
    ```
  - 响应示例：
    ```json
    {
      "recyclable": true,
      "detections": [
        { "class_id": 39, "class_name": "bottle", "confidence": 0.91, "bbox": [x1, y1, x2, y2] }
      ]
    }
    ```
- 其它：
  - GET `/` 返回静态首页 [`frontend/out/index.html`](frontend/out/index.html)
  - GET `/_next/*` 返回静态资源
  - GET `/favicon.ico` 返回 favicon（若存在）

具体实现参见：[backend/app.py](backend/app.py)。

## 配置

- 环境变量（后端读取）：
  - `MODEL_PATH`：YOLO 权重路径，默认 `model/best.pt`
  - `RECYCLABLE_CLASSES`：可回收类名关键词，逗号分隔，默认 `recyclable,可回收,plastic,glass,metal,paper,can,bottle`
- 前端静态导出：[`next.config.ts`](frontend/next.config.ts) 已启用 `output: 'export'`

## 容器部署

构建并运行（Compose 已配置使用 host 网络以兼容部分环境不支持 bridge/veth 的限制）：

```bash
docker compose build
docker compose up -d
# 访问 http://127.0.0.1:8000/
```

- 镜像多阶段构建：前端在 Node 阶段 `pnpm build` 导出到 `frontend/out`，运行阶段使用 `python:3.12-slim` 安装后端依赖并托管静态资源（见 [Dockerfile](Dockerfile)）
- 编排（host 网络、健康检查、模型挂载）：见 [docker-compose.yml](docker-compose.yml)

如宿主机支持默认桥接网络，可移除 `network_mode: host` 并改用 `ports: ["8000:8000"]`。

## 常见问题

- CSS 未生效
  - 确认 [`app/globals.css`](frontend/app/globals.css) 顶部为 `@import "tailwindcss";`
  - PostCSS 使用 `@tailwindcss/postcss` 插件（见 [`postcss.config.mjs`](frontend/postcss.config.mjs)）
  - 前端需先 `pnpm build` 生成 `frontend/out` 再由后端托管
- PyTorch 2.6+ 安全反序列化报错
  - 已在后端对 Ultralytics 自定义类进行 allowlist（`torch.serialization.add_safe_globals(...)`），详见 [backend/app.py](backend/app.py)
- favicon 或 DevTools 探测 404
  - 放置 `frontend/public/favicon.ico` 并重新构建；或后端路由已提供空响应避免噪音

## 许可

本项目依赖 Ultralytics、FastAPI、Next.js、Tailwind 等开源软件，具体许可以各依赖为准。
