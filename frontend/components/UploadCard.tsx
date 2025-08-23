'use client'
import React, { useRef, useState } from 'react'

type Detection = {
  class_id: number
  class_name: string
  confidence: number
  bbox: number[] // [x1,y1,x2,y2]
  // 兼容后端可能返回的扩展字段
  root?: string
  is_recyclable?: boolean
}

type PredictResponse = {
  recyclable?: boolean
  major_category?: string | null
  scores_by_category?: Record<string, number>
  detections?: Detection[]
}

// 提取“顶级分类”（支持半/全角/长破折号），并做别名归一
function extractRootCategory(name: string): string {
  const head = (name?.split(/[-－—]+/)[0] ?? '').trim()
  if (head === '其它垃圾' || head === '其余垃圾') return '其他垃圾'
  if (head === '可回收') return '可回收物'
  return head || name
}

function resizeImageCanvas(img: HTMLImageElement, maxWidth = 640) {
  const ratio = Math.min(1, maxWidth / img.naturalWidth)
  const w = Math.round(img.naturalWidth * ratio)
  const h = Math.round(img.naturalHeight * ratio)
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  const ctx = c.getContext('2d')!
  ctx.drawImage(img, 0, 0, w, h)
  return c
}

function drawHighDPRCanvas(canvas: HTMLCanvasElement, width: number, height: number) {
  const dpr = window.devicePixelRatio || 1
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`
  canvas.width = Math.round(width * dpr)
  canvas.height = Math.round(height * dpr)
  const ctx = canvas.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return ctx
}

export default function UploadCard() {
  const fileRef = useRef<HTMLInputElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [loading, setLoading] = useState(false)
  const [detections, setDetections] = useState<Detection[]>([])
  const [major, setMajor] = useState<string | null>(null)

  async function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setLoading(true)
    const img = new Image()
    img.src = URL.createObjectURL(f)
    await new Promise((r) => (img.onload = r))
    // const tmp = resizeImageCanvas(img, 640)
    // 以占位容器的实际宽度作为缩放上限（回退到 640）
     const targetCssW =
    Math.max(1, Math.floor(containerRef.current?.clientWidth ?? 640))
    const tmp = resizeImageCanvas(img, targetCssW)
    const blob = await new Promise<Blob | null>((res) =>
      tmp.toBlob((b) => res(b), 'image/jpeg', 0.9)
    )
    if (!blob) {
      setLoading(false)
      return
    }

    // draw original image to visible canvas with high DPR
    if (canvasRef.current) {
      const cssW = tmp.width
      const cssH = tmp.height
      const ctx = drawHighDPRCanvas(canvasRef.current, cssW, cssH)
      const imgEl = new Image()
      imgEl.src = tmp.toDataURL('image/jpeg', 0.9)
      await new Promise((r) => (imgEl.onload = r))
      ctx.drawImage(imgEl, 0, 0, cssW, cssH)
    }

    // upload
    const fd = new FormData()
    fd.append('file', blob, 'upload.jpg')
    try {
      const res = await fetch('/predict', { method: 'POST', body: fd })
      if (!res.ok) throw new Error('网络错误')
      const data: PredictResponse = await res.json()
      console.log('predict response:', data)
      const dets: Detection[] = data.detections ?? []
      setDetections(dets)
      // 仅使用后端返回的四大类判定结果，保持与后端一致
      const majorFromServer = (typeof data.major_category === 'string' && data.major_category) ? data.major_category : null
      setMajor(majorFromServer)

      // draw boxes (use CSS pixel coords because ctx is scaled)
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d')!
        ctx.lineWidth = Math.max(2, Math.round(Math.min(canvasRef.current.width, canvasRef.current.height) / (window.devicePixelRatio * 200)))
        ctx.strokeStyle = '#ff4757'
        ctx.font = '12px sans-serif'
        // 将后端 bbox（原图坐标）按缩放比换算到当前可视画布
        const scaleX = tmp.width / img.naturalWidth
        const scaleY = tmp.height / img.naturalHeight
        dets.forEach((d: Detection) => {
          const [x1, y1, x2, y2] = d.bbox
          const sx1 = Math.round(x1 * scaleX)
          const sy1 = Math.round(y1 * scaleY)
          const sx2 = Math.round(x2 * scaleX)
          const sy2 = Math.round(y2 * scaleY)
          ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1)
          const label = `${d.class_name} ${(d.confidence * 100).toFixed(1)}%`
          const tw = ctx.measureText(label).width + 8
          ctx.fillStyle = 'rgba(255,71,87,0.9)'
          const ty = Math.max(0, sy1 - 18)
          ctx.fillRect(sx1, ty, tw, 16)
          ctx.fillStyle = '#fff'
          ctx.fillText(label, sx1 + 4, ty + 12)
        })
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      alert('检测失败：' + msg)
    }
    setLoading(false)
  }

  // 根据检测结果在前端兜底推断四大类（当后端未提供时）
  function deriveMajor(dets: Detection[]): string | null {
    if (!dets.length) return null
    const roots = dets.map(d => extractRootCategory(d.class_name))
    const uniq = Array.from(new Set(roots))
    if (uniq.length === 1) return uniq[0]
    const sums = new Map<string, number>()
    for (let i = 0; i < dets.length; i++) {
      const r = roots[i]
      sums.set(r, (sums.get(r) ?? 0) + (dets[i].confidence || 0))
    }
    let best: string | null = null
    let bestScore = -1
    for (const [k, v] of sums) {
      if (v > bestScore) { best = k; bestScore = v }
    }
    return best
  }

  function majorBadgeClass(m: string) {
    switch (m) {
      case '可回收物': return 'bg-green-100 text-green-800'
      case '厨余垃圾': return 'bg-yellow-100 text-yellow-800'
      case '其他垃圾': return 'bg-gray-100 text-gray-800'
      case '有害垃圾': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-700'
    }
  }

  function reset() {
    setDetections([])
    setMajor(null)
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')!
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
  }

  return (
    <div className="bg-white rounded-xl p-4 mt-4 shadow">
      <div className="flex gap-2">
        <button
          className="px-4 py-2 rounded bg-blue-600 text-white"
          onClick={() => fileRef.current?.click()}
        >
          选择图片
        </button>
        <button className="px-4 py-2 rounded border" onClick={reset}>
          返回开始
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={onPick}
        />
      </div>

      {/* 结果徽章提前到画布上方，避免被大图区域挤到首屏之外 */}
      <div className="mt-3 min-h-[2rem]">
        {major ? (
          <span className={`inline-block px-3 py-1 rounded ${majorBadgeClass(major)}`}>{major}</span>
        ) : loading ? (
          <span className="text-sm text-gray-500">检测中…</span>
        ) : (
          <span className="text-sm text-gray-400">等待检测结果</span>
        )}
      </div>

      <div
        className="mt-2 h-[50dvh] md:h-[60dvh] lg:h-[65dvh] min-h-[240px] max-h-[640px]"
        ref={containerRef}
      >
        <canvas ref={canvasRef} className="w-full h-full rounded bg-gray-100" />
      </div>

      {/* 保留详细结果列表 */}
      <div className="mt-3 text-sm text-gray-700">
        {detections.map((d, i) => (
          <div key={i} className="mb-1">
            {d.class_name} · {(d.confidence * 100).toFixed(1)}%
          </div>
        ))}
      </div>

      {loading && <div className="mt-2 text-sm text-gray-500">检测中…</div>}
    </div>
  )
}
