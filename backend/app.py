import os
import io
from typing import List
from fastapi import FastAPI, File, UploadFile
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

app = FastAPI(title='GC Detector')
# 把前端静态文件挂载到 /static（前端 index.html 我们直接在项目根 frontend 目录）
app.mount('/_next', StaticFiles(directory='frontend/out/_next'), name='next-assets')

# 根路由直接返回前端页面
@app.get('/')
async def index():
    return FileResponse('frontend/out/index.html')

# 延迟加载模型（程序启动时加载）
print('加载模型:', MODEL_PATH)
model = YOLO(MODEL_PATH)
print('模型加载完成')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # 读取并转换为 numpy RGB 图片
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return JSONResponse({'error': '无法读取图片', 'detail': str(e)}, status_code=400)
    img_np = np.array(img)

    # 运行推理（CPU）
    results = model.predict(source=[img_np], device='cpu', imgsz=640, conf=0.25)
    if not results:
        return {'recyclable': False, 'detections': []}

    r = results[0]
    detections = []
    recyclable = False

    # 模型可能包含 names 映射
    names = getattr(model, 'names', {}) or {}

    boxes = getattr(r, 'boxes', None)
    if boxes is None:
        # 未包含 boxes（极端情况）
        return {'recyclable': False, 'detections': []}

    # boxes 是一个 BoxList-like （ultralytics）
    for b in boxes:
        try:
            # 尝试以新 API 提取（tensor）
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0].cpu().numpy())
            cls_id = int(b.cls[0].cpu().numpy())
        except Exception:
            # 回退：尝试属性直接读
            try:
                xyxy = [float(x) for x in b.xyxy]
            except Exception:
                xyxy = [0,0,0,0]
            conf = float(getattr(b, 'conf', 0))
            cls_id = int(getattr(b, 'cls', 0))
        cls_name = str(names.get(cls_id, str(cls_id)))
        bbox = [int(round(x)) for x in xyxy]
        detections.append({'class_id': cls_id, 'class_name': cls_name, 'confidence': conf, 'bbox': bbox})
        low = cls_name.lower()
        if any(token in low for token in RECYCLABLE_TOKENS):
            recyclable = True

    return {'recyclable': recyclable, 'detections': detections}

# 仅当作为独立程序运行时启动（镜像会用 uvicorn 启动）
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.app:app', host='0.0.0.0', port=8000, workers=1)