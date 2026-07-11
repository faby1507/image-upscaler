# 🚀 Real-ESRGAN Image Upscaler (Apple Silicon & Metal)

This script allows you to **upscale single images or entire folders** leveraging the power of Apple Silicon chips (M1, M2, M3, M4, etc.) via the **Metal (MPS)** backend, while also supporting CUDA (NVIDIA) and CPU fallbacks.

---

## 💻 Commands & Usage Examples

After installing the required dependencies (`pip install torch torchvision pillow basicsr realesrgan`), you can run the script directly from your terminal using these practical commands:

### 1. Standard Upscaling (4x, Photo Mode)
Takes an image and creates a 4x enlarged copy named `photo_upscaled.jpg` in the same directory.
```bash
python upscale.py photo.jpg
