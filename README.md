# jundek

一个基于 FastAPI + Ultralytics YOLO 的垃圾分类演示应用：
- 后端提供 `/predict` 推理接口，使用“检测模型”结果聚合为中国垃圾分类四大类（厨余垃圾 / 可回收物 / 其他垃圾 / 有害垃圾），并内置默认类名映射；
- 前端使用 Next.js 15（Turbopack）+ Tailwind CSS v4，静态导出后由后端托管 `/_next` 与首页。

- 后端入口：`backend/app.py`
- 前端配置：`frontend/next.config.ts`
- 前端页面与组件：`frontend/app/page.tsx`、`frontend/components/UploadCard.tsx`
- 依赖清单：`requirements.txt`
- 容器构建：多阶段 `Dockerfile`，编排 `docker-compose.yml`

## 已实现功能

- 图片上传、按容器宽度自适应缩放、高清 DPR 渲染；
- YOLO 推理（检测模型）：解析 boxes，提供可视化框与类别；
- 四大类合并判定：厨余垃圾 / 可回收物 / 其他垃圾 / 有害垃圾；
- 前端静态导出（`output: 'export'`）、后端统一托管 `/_next` 与首页；
- 运行观测：`/__ping` 健康检查、全局响应头 `X-App-Tag` 标记实例；
- 兼容 PyTorch 2.6+ 安全反序列化（允许 Ultralytics 自定义类）。

## 目录结构

- 后端 API 与模型加载：`backend/app.py`
  - 环境变量：
    - `MODEL_PATH`、`DETECT_CONF`、`RECYCLABLE_ROOTS`、`APP_TAG`
    - `CLASS_ROOT_MAP`（JSON，可覆盖“类名 → 四大类”默认映射）
    - `CLASS_ROOT_KEYWORDS`（JSON，可覆盖关键词启发式）
  - 路由：
    - `GET /`：静态首页
    - `POST /predict`：图片推理
    - `GET /_next/*`：前端静态资源
    - `GET /__ping`：健康检查（返回 `ok/tag/model_path`）
- 前端应用（静态导出）：`frontend`
  - `next.config.ts`（`output: 'export'`）
  - `app/globals.css`（Tailwind v4：`@import "tailwindcss";`）
  - `postcss.config.mjs`（`@tailwindcss/postcss`）
- 容器与编排：`Dockerfile`、`docker-compose.yml`

## 本地开发

前置：Linux、Python 3.12、Node.js 20+、pnpm（`corepack enable`）。

1) 安装 Python 依赖并准备模型
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 将模型权重放到 model/best.pt（或设置 MODEL_PATH）
```

2) 构建前端静态资源
```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build   # 生成 frontend/out
cd ..
```

3) 启动后端
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
```

4) 打开浏览器访问 `http://127.0.0.1:8000/`

说明：
- 若需前端独立开发：`pnpm dev`（端口 3000），请把前端请求 API 指向 `http://127.0.0.1:8000`（或为后端开启 CORS）。

## API

### POST `/predict`
接收表单文件字段 `file`（image/*）。

请求（示例）：
```bash
curl -F "file=@/path/to/image.jpg" http://127.0.0.1:8000/predict
```

响应：
```json
{
  "recyclable": true,
  "major_category": "可回收物",        // 或 null（无明显结果）
  "scores_by_category": {               // 四大类置信度汇总（基于检测框置信度）
    "厨余垃圾": 0.12,
    "可回收物": 0.76,
    "其他垃圾": 0.09,
    "有害垃圾": 0.03
  },
  "detections": [                       // 仅超过展示阈值（DETECT_CONF）才包含
    {
      "class_id": 15,
      "class_name": "Bottle",         // 原始模型类名
      "confidence": 0.88,
      "bbox": [x1, y1, x2, y2],
      "root": "可回收物",            // 中文四大类
      "is_recyclable": true
    }
  ]
}
```

响应头：全局包含 `X-App-Tag: <实例标识>`，便于排障。

### GET `/__ping`
健康检查与运维：
```json
{ "ok": true, "tag": "ts-...", "model_path": "model/best.pt" }
```

## 配置（环境变量）

- `MODEL_PATH`：模型权重路径（默认 `model/best.pt`）。
- `DETECT_CONF`：检测框“展示阈值”，仅影响返回 `detections`，不影响大类判定（默认 `0.01`）。
- `RECYCLABLE_ROOTS`：被视为“可回收物”的顶级类名集合（默认 `可回收物`）。
- `APP_TAG`：实例标记字符串（默认启动时间戳）。
- `CLASS_ROOT_MAP`：覆盖类名到四大类的映射（JSON 字符串），示例：
  - `{"Plastic bag":"其他垃圾","Tin can":"可回收物"}`
- `CLASS_ROOT_KEYWORDS`：覆盖关键词启发式（JSON 字符串）。

说明：
- 已内置对 30 个类别名的默认映射，通常无需配置。

## 容器部署（Docker/Compose）

构建并运行（Compose 当前使用 host 网络以兼容某些环境不支持 veth）：
```bash
docker compose build
docker compose up -d
# 访问 http://127.0.0.1:8000/
```

关键点：
- 多阶段镜像：前端在 Node 阶段 `pnpm build` 导出 `frontend/out`；运行阶段基于 `python:3.12-slim` 托管静态与接口（见 `Dockerfile`）。
- 健康检查建议用 `__ping`：
  ```yaml
  healthcheck:
    test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:8000/__ping || exit 1"]
    interval: 30s
    timeout: 5s
    retries: 3
  ```
- 覆盖映射示例：
  ```yaml
  environment:
    - MODEL_PATH=/app/model/best.pt
    - CLASS_ROOT_MAP={"Plastic bag":"其他垃圾","Tin can":"可回收物"}
    - DETECT_CONF=0.01
    - APP_TAG=compose-prod
  ```
- 若宿主支持 bridge 网络，可去掉 `network_mode: host` 并添加 `ports: ["8000:8000"]`。

## 故障排查（速查）

- 页面只显示 HTML、无样式：
  - 确认 `frontend/app/globals.css` 顶部为 `@import "tailwindcss";`；
  - `postcss.config.mjs` 使用 `@tailwindcss/postcss`；
  - 先 `pnpm build` 生成 `frontend/out` 再由后端托管。
- `/predict` 返回空：
  - 查看后端日志：应打印 `boxes 数组形状 ...` 与 `分类总分 ...`；
  - Network 响应头应含 `X-App-Tag` 且与 `GET /__ping` 一致；
  - 若日志缺失，多半命中了旧进程或不同端口，请确保只运行一个后端实例。
- PyTorch 2.6+ 反序列化错误：
  - 已通过 `torch.serialization.add_safe_globals(...)` 允许所需 Ultralytics 类型。

## 默认类名映射（30 类 → 四大类）

- 厨余垃圾：Banana, Apple, Orange, Tomato, Carrot, Cucumber, Potato, Bread, Cake, Pizza, Hamburger, Chicken, Fish, Food, Fast food, Pasta, Pastry, Snack, Candy
- 可回收物：Tin can, Bottle, Milk, Facial tissue holder, Soap dispenser
- 其他垃圾：Toothbrush, Drinking straw, Plastic bag, Toilet paper, Paper towel
- 有害垃圾：Light bulb

如需调整，可通过 `CLASS_ROOT_MAP` 环境变量覆盖单项映射。

## 许可

本项目依赖 Ultralytics、FastAPI、Next.js、Tailwind 等开源软件，具体许可以各依赖为准。
# jundek

一个基于 FastAPI + Ultralytics YOLO 的垃圾分类演示应用：
- 后端负责加载模型并提供 `/predict` 推理接口；支持检测模型与分类模型，按“四大类”聚合；
- 前端使用 Next.js 15（Turbopack）+ Tailwind CSS v4，静态导出后由后端托管 `/_next` 与首页。

- 后端入口：`backend/app.py`
- 前端配置：`frontend/next.config.ts`
- 前端页面与组件：`frontend/app/page.tsx`、`frontend/components/UploadCard.tsx`
- 依赖清单：`requirements.txt`
- 容器构建：多阶段 `Dockerfile`，编排 `docker-compose.yml`

## 已实现功能

- 图片上传、按容器宽度自适应缩放、高清 DPR 渲染；
- YOLO 推理：
  - 检测模型：解析 boxes，提供可视化框与类别；
  - 分类模型：解析 probs 向量，按大类聚合（支持多种聚合策略）；
- 四大类合并判定：厨余垃圾 / 可回收物 / 其他垃圾 / 有害垃圾；
- 前端静态导出（`output: 'export'`）、后端统一托管 `/_next` 与首页；
- 运行观测：`/__ping` 健康检查、全局响应头 `X-App-Tag` 标记实例；
- 兼容 PyTorch 2.6+ 安全反序列化（允许 Ultralytics 自定义类）。

## 目录结构

- 后端 API 与模型加载：`backend/app.py`
  - 环境变量：`MODEL_PATH`、`DETECT_CONF`、`RECYCLABLE_ROOTS`、`CLASS_AGGREGATION`、`CLASS_TOPK`、`CLASS_MIN_PROB`、`APP_TAG` 等；
  - 路由：
    - `GET /`：静态首页
    - `POST /predict`：图片推理
    - `GET /_next/*`：前端静态资源
    - `GET /__ping`：健康检查（返回 `ok/tag/model_path`）
- 前端应用（静态导出）：`frontend`
  - `next.config.ts`（`output: 'export'`）
  - `app/globals.css`（Tailwind v4：`@import "tailwindcss";`）
  - `postcss.config.mjs`（`@tailwindcss/postcss`）
- 容器与编排：`Dockerfile`、`docker-compose.yml`

## 本地开发

前置：Linux、Python 3.12、Node.js 20+、pnpm（`corepack enable`）。

1) 安装 Python 依赖并准备模型
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 将模型权重放到 model/best.pt（或设置 MODEL_PATH）
```

2) 构建前端静态资源
```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build   # 生成 frontend/out
cd ..
```

3) 启动后端
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
```

4) 打开浏览器访问 `http://127.0.0.1:8000/`

说明：
- 若需前端独立开发：`pnpm dev`（端口 3000），请把前端请求 API 指向 `http://127.0.0.1:8000`（或为后端开启 CORS）。

## API

### POST `/predict`
接收表单文件字段 `file`（image/*）。

请求（示例）：
```bash
curl -F "file=@/path/to/image.jpg" http://127.0.0.1:8000/predict
```

响应（检测或分类均统一返回结构）：
```json
{
  "recyclable": true,
  "major_category": "可回收物",        // 或 null（无明显结果）
  "scores_by_category": {               // 四大类置信度汇总
    "厨余垃圾": 0.12,
    "可回收物": 0.76,
    "其他垃圾": 0.09,
    "有害垃圾": 0.03
  },
  "detections": [                       // 仅检测模型且超过展示阈值才包含
    {
      "class_id": 123,
      "class_name": "可回收物-饮料瓶",
      "confidence": 0.88,
      "bbox": [x1, y1, x2, y2],
      "root": "可回收物",
      "is_recyclable": true
    }
  ]
}
```

响应头：全局包含 `X-App-Tag: <实例标识>`，便于排障。

### GET `/__ping`
健康检查与运维：
```json
{ "ok": true, "tag": "ts-...", "model_path": "model/best.pt" }
```

## 配置（环境变量）

- `MODEL_PATH`：模型权重路径（默认 `model/best.pt`）。
- `DETECT_CONF`：检测框“展示阈值”，仅影响返回 `detections`，不影响大类判定（默认 `0.01`）。
- `RECYCLABLE_ROOTS`：被视为“可回收物”的顶级类名集合（默认 `可回收物`）。
- 分类聚合（仅分类模型有效）：
  - `CLASS_AGGREGATION`：`sum_all | topk_sum | top1_max | normalized_sum`（默认 `sum_all`）。
  - `CLASS_TOPK`：`topk_sum` 模式下的 K 值（默认 `5`）。
  - `CLASS_MIN_PROB`：忽略低于此阈值的细分类概率（默认 `0.0`）。
- `APP_TAG`：实例标记字符串（默认启动时间戳）。

说明：
- `sum_all` 会偏向细分类数量更多的大类；若希望更贴近“Top项直觉”，推荐 `CLASS_AGGREGATION=topk_sum`，可配 `CLASS_TOPK=5`、`CLASS_MIN_PROB=0.01`。

## 容器部署（Docker/Compose）

构建并运行（Compose 当前使用 host 网络以兼容某些环境不支持 veth）：
```bash
docker compose build
docker compose up -d
# 访问 http://127.0.0.1:8000/
```

关键点：
- 多阶段镜像：前端在 Node 阶段 `pnpm build` 导出 `frontend/out`；运行阶段基于 `python:3.12-slim` 托管静态与接口（见 `Dockerfile`）。
- 健康检查建议用 `__ping`：
  ```yaml
  healthcheck:
    test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:8000/__ping || exit 1"]
    interval: 30s
    timeout: 5s
    retries: 3
  ```
- 可暴露分类聚合参数，示例：
  ```yaml
  environment:
    - MODEL_PATH=/app/model/best.pt
    - CLASS_AGGREGATION=topk_sum
    - CLASS_TOPK=5
    - CLASS_MIN_PROB=0.01
    - DETECT_CONF=0.01
    - APP_TAG=compose-prod
  ```
- 若宿主支持 bridge 网络，可去掉 `network_mode: host` 并添加 `ports: ["8000:8000"]`。

## 故障排查（速查）

- 页面只显示 HTML、无样式：
  - 确认 `frontend/app/globals.css` 顶部为 `@import "tailwindcss";`；
  - `postcss.config.mjs` 使用 `@tailwindcss/postcss`；
  - 先 `pnpm build` 生成 `frontend/out` 再由后端托管。
- `/predict` 返回空：
  - 查看后端日志：应打印 `boxes 数组形状 ...` 或 `classification 向量形状 ...` 与 `分类总分 ...`；
  - Network 响应头应含 `X-App-Tag` 且与 `GET /__ping` 一致；
  - 若日志缺失，多半命中了旧进程或不同端口，请确保只运行一个后端实例。
- PyTorch 2.6+ 反序列化错误：
  - 已通过 `torch.serialization.add_safe_globals(...)` 允许所需 Ultralytics 类型。

## 许可

本项目依赖 Ultralytics、FastAPI、Next.js、Tailwind 等开源软件，具体许可以各依赖为准。
