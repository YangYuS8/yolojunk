import os
import io
import re
import time
from collections import defaultdict
from typing import List
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Concat

torch.serialization.add_safe_globals([
    DetectionModel,
    Bottleneck,
    Concat,
])

MODEL_PATH = os.getenv('MODEL_PATH', 'model/best.pt')
# 识别到这些类名之一时，认为是可回收；可用逗号分隔的关键词（小写匹配）
RECYCLABLE_CLASSES = os.getenv('RECYCLABLE_CLASSES', 'recyclable,可回收,plastic,glass,metal,paper,can,bottle')
RECYCLABLE_TOKENS = set([s.strip().lower() for s in RECYCLABLE_CLASSES.split(',') if s.strip()])

# 顶级分类判定（适配“可回收物-xxx / 其他垃圾-xxx / 厨余垃圾-xxx / 有害垃圾-xxx”）
# 可回收的顶级分类（逗号分隔，可通过环境变量覆盖）
RECYCLABLE_ROOTS = set(s.strip() for s in os.getenv('RECYCLABLE_ROOTS', '可回收物').split(',') if s.strip())
# 置信度阈值（默认 0.25；可通过环境变量 DETECT_CONF 调整，例如 0.1 或 0.05）
# 展示阈值（仅用于返回给前端画框；最终大类判定不使用阈值，只看置信度总和）
DETECT_CONF = float(os.getenv('DETECT_CONF', '0.01'))
# 分类聚合策略：sum_all | topk_sum | top1_max | normalized_sum
CLASS_AGGREGATION = os.getenv('CLASS_AGGREGATION', 'sum_all').strip().lower()
CLASS_TOPK = int(os.getenv('CLASS_TOPK', '5'))
CLASS_MIN_PROB = float(os.getenv('CLASS_MIN_PROB', '0.0'))
# 支持半/全角/长破折号
DASH_PATTERN = re.compile(r'[-－—]+')
# 顶级分类别名归一
ALT_ROOT_MAP = {
    '其它垃圾': '其他垃圾',
    '其余垃圾': '其他垃圾',
    '可回收': '可回收物',
}

def extract_root_category(name: str) -> str:
    base = name.strip()
    head = DASH_PATTERN.split(base, maxsplit=1)[0].strip()
    return ALT_ROOT_MAP.get(head, head)

def is_recyclable_name(name: str) -> bool:
    # 优先按顶级分类判断
    root = extract_root_category(name)
    if root in RECYCLABLE_ROOTS:
        return True
    # 兼容旧子串匹配（若配置了 RECYCLABLE_CLASSES）
    if RECYCLABLE_TOKENS:
        low = name.lower()
        return any(tok in low for tok in RECYCLABLE_TOKENS)
    return False

app = FastAPI(title='GC Detector')
# 把前端静态文件挂载到 /static（前端 index.html 我们直接在项目根 frontend 目录）
# app.mount('/static', StaticFiles(directory='frontend'), name='static')
app.mount('/_next', StaticFiles(directory='frontend/out/_next'), name='next-assets')

# 根路由直接返回前端页面
@app.get('/')
async def index():
    return FileResponse('frontend/out/index.html')

# 统一给所有响应加上实例标识，便于前端或 curl 侧确认命中的服务实例
@app.middleware('http')
async def add_app_tag_header(request, call_next):
    response = await call_next(request)
    try:
        response.headers['X-App-Tag'] = APP_TAG
    except Exception:
        pass
    return response

# 实例标识（用于确认命中的是哪一版后端）
APP_TAG = os.getenv('APP_TAG') or f"ts-{int(time.time())}"

# 延迟加载模型（程序启动时加载）
print('APP_TAG:', APP_TAG, flush=True)
print('加载模型:', MODEL_PATH, flush=True)
model = YOLO(MODEL_PATH)
print('模型加载完成', flush=True)

@app.get('/__ping')
async def ping():
    return {
        'ok': True,
        'tag': APP_TAG,
        'model_path': MODEL_PATH,
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # 读取并转换为 numpy RGB 图片
    contents = await file.read()
    try:
        print('predict 开始, 收到字节:', len(contents), 'tag:', APP_TAG, flush=True)
    except Exception:
        pass
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return JSONResponse({'error': '无法读取图片', 'detail': str(e)}, status_code=400)
    img_np = np.array(img)

    # 运行推理（CPU）
    results = model.predict(source=[img_np], device='cpu', imgsz=640, conf=DETECT_CONF)
    if not results:
        return {
            'recyclable': False,
            'major_category': None,
            'scores_by_category': {},
            'detections': []
        }

    r = results[0]
    detections = []
    # 用于大类决策的累加（不受 DETECT_CONF 过滤）
    sums_all = defaultdict(float)

    # 模型可能包含 names 映射
    names = getattr(r, 'names', None) or getattr(model, 'names', {}) or {}
    try:
        if isinstance(names, dict):
            print('names 类型: dict, 大小:', len(names), flush=True)
        elif isinstance(names, (list, tuple)):
            print('names 类型: list/tuple, 大小:', len(names), flush=True)
        else:
            print('names 类型: 其他', flush=True)
    except Exception:
        pass

    boxes = getattr(r, 'boxes', None)
    has_data_attr = False
    try:
        has_data_attr = hasattr(boxes, 'data')
    except Exception:
        has_data_attr = False
    try:
        print('boxes 是否存在:', boxes is not None, 'has data:', has_data_attr, 'type:', type(boxes).__name__ if boxes is not None else None, flush=True)
    except Exception:
        pass

    # 分类分支（适配分类模型）：当没有检测框或 boxes 无 data 时，使用 r.probs 进行大类判定
    if (boxes is None) or (not has_data_attr):
        probs_obj = getattr(r, 'probs', None)
        if probs_obj is not None:
            # 取概率向量
            try:
                vec = probs_obj.data if hasattr(probs_obj, 'data') else probs_obj
                vec = vec.cpu().numpy() if hasattr(vec, 'cpu') else np.array(vec)
            except Exception as e:
                print('读取分类 probs 失败:', repr(e), flush=True)
                vec = np.zeros((0,), dtype=float)
            try:
                print('classification 向量形状:', tuple(vec.shape), flush=True)
            except Exception:
                pass

            # names 可能为 dict 或 list/tuple
            def _name_of(i: int) -> str:
                if isinstance(names, dict):
                    return str(names.get(i, str(i)))
                elif isinstance(names, (list, tuple)):
                    return str(names[i]) if 0 <= i < len(names) else str(i)
                else:
                    return str(i)

            # 预计算每个类的根类别
            indices = list(range(len(vec)))
            class_roots = {i: extract_root_category(_name_of(i)) for i in indices}

            # 统计每个大类包含的细分类数量（用于 normalized_sum）
            root_class_count = defaultdict(int)
            for i in indices:
                root_class_count[class_roots[i]] += 1

            method = CLASS_AGGREGATION
            topk = max(1, CLASS_TOPK)
            min_p = max(0.0, CLASS_MIN_PROB)

            if method == 'top1_max':
                # 取全体中概率最大单类所属大类
                best_idx = int(np.argmax(vec)) if len(vec) else -1
                if best_idx >= 0:
                    root = class_roots[best_idx]
                    sums_all[root] += float(vec[best_idx])
            elif method == 'topk_sum':
                # 仅用全体中前 K 个最高概率类累加
                if len(vec):
                    top_idx = np.argsort(vec)[-topk:][::-1]
                    for i in top_idx:
                        p = float(vec[i])
                        if p < min_p:
                            continue
                        sums_all[class_roots[int(i)]] += p
            elif method == 'normalized_sum':
                # 对每个类的概率按大类的“细分类数量”归一后再求和，降低类目数量差异的偏置
                for i in indices:
                    p = float(vec[i])
                    if p < min_p:
                        continue
                    root = class_roots[i]
                    denom = float(root_class_count.get(root, 1))
                    sums_all[root] += p / denom
            else:
                # 默认：对所有类的概率直接求和（可能偏向细分类更多的大类）
                for i in indices:
                    p = float(vec[i])
                    if p < min_p:
                        continue
                    sums_all[class_roots[i]] += p

            known_roots = {'厨余垃圾', '可回收物', '其他垃圾', '有害垃圾'}
            candidates = {k: v for k, v in sums_all.items() if k in known_roots} or dict(sums_all)
            major_category = max(candidates.items(), key=lambda kv: kv[1])[0] if candidates else None
            scores_by_category = candidates
            recyclable = (major_category == '可回收物')

            try:
                print('分类聚合方法:', method, 'topk:', topk, 'min_p:', min_p, flush=True)
                print('分类总分(CLASSIFICATION):', {k: round(v, 4) for k, v in scores_by_category.items()}, '=>', major_category, flush=True)
            except Exception:
                pass

            payload = {
                'recyclable': recyclable,
                'major_category': major_category,
                'scores_by_category': scores_by_category,
                'detections': []
            }
            return JSONResponse(payload, headers={'X-App-Tag': APP_TAG})
        # 分类与检测都不可用
        return {
            'recyclable': False,
            'major_category': None,
            'scores_by_category': {},
            'detections': []
        }

    # 稳健读取 Ultralytics Boxes：优先用 boxes.data (N,6) => [x1,y1,x2,y2,conf,cls]
    arr = None
    try:
        data = boxes.data
        arr = data.cpu().numpy() if hasattr(data, 'cpu') else np.array(data)
    except Exception as e:
        # data 读取失败，退回分别取属性
        try:
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy().astype(int)
            arr = np.concatenate([xyxys, confs.reshape(-1,1), clss.reshape(-1,1)], axis=1)
        except Exception as e2:
            print('读取 boxes 失败:', repr(e), '| fallback 失败:', repr(e2), flush=True)
            arr = np.zeros((0, 6), dtype=float)

    # 兼容极端情况：如果 data 维度异常，退回分别取属性
    if arr is None or arr.ndim != 2 or arr.shape[1] < 6:
        try:
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy().astype(int)
            arr = np.concatenate([xyxys, confs.reshape(-1,1), clss.reshape(-1,1)], axis=1)
        except Exception as e3:
            print('boxes 维度异常且 fallback 失败:', repr(e3), flush=True)
            arr = np.zeros((0, 6), dtype=float)

    # 调试：打印当前 boxes 行列信息
    try:
        print('boxes 数组形状:', tuple(arr.shape), flush=True)
    except Exception:
        pass

    n = int(arr.shape[0]) if arr is not None else 0
    for i in range(n):
        x1, y1, x2, y2, conf, cls_id_f = arr[i, :6].tolist()
        conf = float(conf)
        cls_id = int(cls_id_f)
        # 兼容 names 为 dict 或 list/tuple
        if isinstance(names, dict):
            cls_name = str(names.get(cls_id, str(cls_id)))
        elif isinstance(names, (list, tuple)):
            if 0 <= cls_id < len(names):
                cls_name = str(names[cls_id])
            else:
                cls_name = str(cls_id)
        else:
            cls_name = str(cls_id)
        root = extract_root_category(cls_name)
        # 参与四大类总分累加（不受阈值影响）
        sums_all[root] += conf
        # 仅当高于展示阈值时返回给前端画框（用于可视化，非判定依据）
        if conf >= DETECT_CONF:
            detections.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'root': root,
                'is_recyclable': (root in RECYCLABLE_ROOTS),
                'confidence': conf,
                'bbox': [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            })

    # 基于四大类进行最终决策：只比较四大类的置信度之和，取总和最大的作为判定
    known_roots = {'厨余垃圾', '可回收物', '其他垃圾', '有害垃圾'}
    candidates = {k: v for k, v in sums_all.items() if k in known_roots} or dict(sums_all)
    major_category = max(candidates.items(), key=lambda kv: kv[1])[0] if candidates else None
    scores_by_category = candidates
    recyclable = (major_category == '可回收物')

    # 调试日志（可临时保留，定位问题后删除）
    try:
        print('分类总分:', {k: round(v, 4) for k, v in scores_by_category.items()}, '=>', major_category, flush=True)
    except Exception:
        pass

    payload = {
        'recyclable': recyclable,
        'major_category': major_category,
        'scores_by_category': scores_by_category,
        'detections': detections
    }
    # 带上实例标识，便于在前端 Network 面板验证是否命中最新服务
    return JSONResponse(payload, headers={'X-App-Tag': APP_TAG})

@app.get('/favicon.ico')
async def favicon():
    p = 'frontend/out/favicon.ico'
    if os.path.exists(p):
        return FileResponse(p)
    # 没有 favicon 时静默返回 204，避免报错日志
    return Response(status_code=204)

@app.get('/.well-known/appspecific/com.chrome.devtools.json')
async def chrome_devtools_probe():
    # 204 No Content：不得包含响应体
    return Response(status_code=204)

# 仅当作为独立程序运行时启动（镜像会用 uvicorn 启动）
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.app:app', host='0.0.0.0', port=8000, workers=1)