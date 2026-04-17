# 🚀 Nero-Quantizer: Surgical LLM Compression

Nero-Quantizer is a hybrid optimization toolkit that combines **Singular Value Decomposition (SVD)** and **INT8 Quantization** to reduce LLM memory footprint without destroying model logic.

## 📊 Performance Benchmark
I used this engine to compress **Qwen2.5-1.5B**, achieving significant VRAM savings:
- **Original Size:** 3.09 GB
- **Nero-Compressed Size:** 1.74 GB
- **Total VRAM Saved:** **39.32%** 📉

## 📦 Try the Model
The compressed weights are hosted on Hugging Face:
👉 **[Nero-Qwen2.5-1.5B-Surgical](https://huggingface.co/neroai14/Nero-Qwen2.5-1.5B-Surgical)**

## 🛠️ How it Works
The Nero Engine applies a surgical approach to model pruning:
1. **Rank Decomposition:** It uses the "Elbow Method" to find low-rank approximations of large MLP layers.
2. **Hybrid Quantization:** It applies INT8 quantization to the SVD-cleaned weights.
3. **Logic Protection:** Sensitive layers (Attention & LM Head) are automatically skipped to maintain reasoning capabilities.

## 🚀 Quick Start
```bash
git clone [https://github.com/yassin607/Nero-Quantizer](https://github.com/yassin607/Nero-Quantizer)
cd Nero-Quantizer
pip install -r requirements.txt
python nero_svd_engine.py
