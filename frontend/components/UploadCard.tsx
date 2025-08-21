'use client'
import React, { useRef, useState } from 'react'

type Detection = {
  class_id: number
  class_name: string
  confidence: number
  bbox: number[] // [x1,y1,x2,y2]
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
  const [loading, setLoading] = useState(false)
  const [detections, setDetections] = useState<Detection[]>([])
  const [recyclable, setRecyclable] = useState<boolean | null>(null)

  async function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setLoading(true)
    const img = new Image()
    img.src = URL.createObjectURL(f)
    await new Promise((r) => (img.onload = r))
    const tmp = resizeImageCanvas(img, 640)
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
      const data = await res.json()
      setDetections(data.detections ?? [])
      setRecyclable(Boolean(data.recyclable))

      // draw boxes (use CSS pixel coords because ctx is scaled)
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d')!
        ctx.lineWidth = Math.max(2, Math.round(Math.min(canvasRef.current.width, canvasRef.current.height) / (window.devicePixelRatio * 200)))
        ctx.strokeStyle = '#ff4757'
        ctx.font = '12px sans-serif'
        const dets = data.detections ?? [];
        dets.forEach((d: Detection) => {
          const [x1, y1, x2, y2] = d.bbox
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
          const label = `${d.class_name} ${(d.confidence * 100).toFixed(1)}%`
          const tw = ctx.measureText(label).width + 8
          ctx.fillStyle = 'rgba(255,71,87,0.9)'
          ctx.fillRect(x1, Math.max(0, y1 - 18), tw, 16)
          ctx.fillStyle = '#fff'
          ctx.fillText(label, x1 + 4, Math.max(0, y1 - 5))
        })
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      alert('检测失败：' + msg)
    }
    setLoading(false)
  }

  function reset() {
    setDetections([])
    setRecyclable(null)
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
          拍照/选图
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

      <div className="mt-4">
        <canvas ref={canvasRef} className="w-full rounded bg-gray-100" />
      </div>

      <div className="mt-3">
        {recyclable === null ? null : (
          <div className={`inline-block px-3 py-1 rounded ${recyclable ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {recyclable ? '可回收' : '不可回收'}
          </div>
        )}
      </div>

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
