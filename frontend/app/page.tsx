'use client'
import UploadCard from '../components/UploadCard'

export default function Page() {
  return (
    <main className="min-h-screen bg-gray-50 p-safe">
      <div className="max-w-xl mx-auto p-4">
        <h1 className="text-xl font-semibold">垃圾回收检测</h1>
        <p className="text-sm text-gray-500 mt-1">移动端优先，拍照或上传图片检测是否可回收</p>

        <div className="mt-4">
          <UploadCard />
        </div>

        <footer className="text-center text-xs text-gray-500 mt-6">轻量版 · CPU-only</footer>
      </div>
    </main>
  )
}
