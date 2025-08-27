import os
import io
import re
import json
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

# 四大类集合（用于过滤与最终判定）
KNOWN_ROOTS = {'厨余垃圾', '可回收物', '其他垃圾', '有害垃圾'}

def _load_json_env(name: str, default):
    val = os.getenv(name)
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        try:
            # 容错：允许使用单引号的 JSON 风格
            return json.loads(val.replace("'", '"'))
        except Exception:
            return default

# 显式类名映射默认值（基于 doc/classname.txt 的 30 类）
_DEFAULT_CLASS_MAP = {
    # 厨余垃圾
    'Banana': '厨余垃圾',
    'Apple': '厨余垃圾',
    'Orange': '厨余垃圾',
    'Tomato': '厨余垃圾',
    'Carrot': '厨余垃圾',
    'Cucumber': '厨余垃圾',
    'Potato': '厨余垃圾',
    'Bread': '厨余垃圾',
    'Cake': '厨余垃圾',
    'Pizza': '厨余垃圾',
    'Hamburger': '厨余垃圾',
    'Chicken': '厨余垃圾',
    'Fish': '厨余垃圾',
    'Food': '厨余垃圾',
    'Fast food': '厨余垃圾',
    'Pasta': '厨余垃圾',
    'Pastry': '厨余垃圾',
    'Snack': '厨余垃圾',
    'Candy': '厨余垃圾',
    # 可回收物
    'Tin can': '可回收物',
    'Bottle': '可回收物',
    'Milk': '可回收物',
    'Facial tissue holder': '可回收物',
    'Soap dispenser': '可回收物',
    # 其他垃圾
    'Toothbrush': '其他垃圾',
    'Drinking straw': '其他垃圾',
    'Plastic bag': '其他垃圾',
    'Toilet paper': '其他垃圾',
    'Paper towel': '其他垃圾',
    # 有害垃圾
    'Light bulb': '有害垃圾',
}

# 显式类名映射：可通过环境变量覆盖上述默认值
CLASS_ROOT_MAP = _load_json_env('CLASS_ROOT_MAP', _DEFAULT_CLASS_MAP)

# 关键词启发式（优先级低于 CLASS_ROOT_MAP，可通过环境变量覆盖）
_DEFAULT_KEYWORDS = {
    '可回收物': [
        'recyclable', 'recycle', 'plastic', 'glass', 'metal', 'paper', 'cardboard',
        'can', 'bottle', 'jar', 'aluminum', 'tin', 'steel', 'carton', 'magazine', 'newspaper',
        '可回收', '塑料', '玻璃', '金属', '纸', '易拉罐', '瓶', '纸箱', '报纸', '杂志'
    ],
    '厨余垃圾': [
        'kitchen', 'food', 'organic', 'leftover', 'vegetable', 'fruit', 'bone', 'peel', 'shell',
        '厨余', '餐厨', '食物', '剩饭', '剩菜', '果皮', '菜叶', '骨头', '蛋壳', '果核'
    ],
    '有害垃圾': [
        'hazard', 'hazardous', 'harmful', 'toxic', 'battery', 'cell', 'accumulator', 'paint', 'solvent', 'mercury',
        '医废', '药', '废药', '电池', '纽扣电池', '蓄电池', '油漆', '溶剂', '汞', '温度计', '杀虫剂'
    ],
    '其他垃圾': [
        'other', 'residual', 'dry', 'trash', 'misc', 'general', 'rest', 'napkin', 'cigarette', 'ash', 'ceramic',
        '其他', '其它', '其他垃圾', '残余', '干垃圾', '烟蒂', '尘土', '陶瓷', '污染', '不可回收'
    ],
}
CLASS_ROOT_KEYWORDS = _load_json_env('CLASS_ROOT_KEYWORDS', _DEFAULT_KEYWORDS)

def normalize_root_name(root: str) -> str:
    if not root:
        return root
    r = ALT_ROOT_MAP.get(root.strip(), root.strip())
    # 若已为四大类之一，直接返回
    if r in KNOWN_ROOTS:
        return r
    # 英文/别名规范化
    low = r.lower()
    if any(k in low for k in ['recyclable', 'recycle']):
        return '可回收物'
    if any(k in low for k in ['kitchen', 'food', 'organic']):
        return '厨余垃圾'
    if any(k in low for k in ['hazard', 'toxic', 'harmful']):
        return '有害垃圾'
    if any(k in low for k in ['other', 'residual', 'dry', 'general']):
        return '其他垃圾'
    return r

def map_name_to_root(name: str) -> str:
    """将模型类名映射到四大类。
    优先级：
    1) 可解析的顶级前缀（中文：'可回收物-xxx' 等）
    2) 显式映射 CLASS_ROOT_MAP
    3) 关键词启发式 CLASS_ROOT_KEYWORDS
    4) 回退到 extract_root_category(name)
    """
    raw = str(name or '').strip()
    if not raw:
        return raw
    # 1) 若形如 “可回收物-xxx”，直接取前缀
    head = DASH_PATTERN.split(raw, maxsplit=1)[0].strip()
    head_norm = normalize_root_name(head)
    if head_norm in KNOWN_ROOTS:
        return head_norm

    # 2) 显式映射（精确匹配，大小写不敏感）
    low = raw.lower()
    for k, v in CLASS_ROOT_MAP.items():
        if low == str(k).lower():
            return normalize_root_name(str(v))

    # 3) 关键词启发式（出现即命中；若多类命中，取首次命中顺序的类别）
    for root, kws in CLASS_ROOT_KEYWORDS.items():
        try:
            for kw in kws:
                if kw and str(kw).lower() in low:
                    return normalize_root_name(root)
        except Exception:
            continue

    # 4) 回退：尝试按破折号前缀归一
    fallback = normalize_root_name(extract_root_category(raw))
    return fallback

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

    # 仅检测分支：若无 boxes，则视为无检出
    # 下方统一按 boxes 读取与聚合

    # 稳健读取 Ultralytics Boxes：优先用 boxes.data (N,6) => [x1,y1,x2,y2,conf,cls]
    arr = None
    if boxes is None:
        arr = np.zeros((0, 6), dtype=float)
    else:
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
        root = map_name_to_root(cls_name)
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
    candidates = {k: v for k, v in sums_all.items() if k in KNOWN_ROOTS} or dict(sums_all)
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

@app.get('/upload.svg')
async def file_svg():
    p = 'frontend/out/upload.svg'
    if os.path.exists(p):
        return FileResponse(p, media_type='image/svg+xml')
    return Response(status_code=404)

@app.get('/.well-known/appspecific/com.chrome.devtools.json')
async def chrome_devtools_probe():
    # 204 No Content：不得包含响应体
    return Response(status_code=204)

# 仅当作为独立程序运行时启动（镜像会用 uvicorn 启动）
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.app:app', host='0.0.0.0', port=8000, workers=1)