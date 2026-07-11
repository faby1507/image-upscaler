# image-upscaler for Mac (apple silicon) users
# 🚀 Real-ESRGAN Image Upscaler (Apple Silicon & Metal)

Uno script Python leggero ed efficiente per eseguire l'**upscaling di immagini singole o intere cartelle** utilizzando l'algoritmo **Real-ESRGAN**. 

Il punto forte di questo script è l'ottimizzazione per gli utenti Mac: sfrutta nativamente il backend **Metal (MPS)** dei chip Apple Silicon (M1, M2, M3, M4, ecc.), garantendo un'elaborazione rapida senza gravare esclusivamente sulla CPU. Supporta comunque l'accelerazione **CUDA** per GPU NVIDIA e il fallback su **CPU**.

---

## ✨ Caratteristiche principali

* 🍏 **Supporto Apple Silicon:** Rilevamento automatico e accelerazione hardware tramite `torch.backends.mps`.
* ⚡ **Pronto all'uso:** Scarica in autonomia i pesi dei modelli pre-addestrati ufficiali alla prima esecuzione.
* 📷 **Doppia Modalità:** Ottimizzato sia per fotografie reali (`photo`) che per illustrazioni/disegni (`anime`).
* 🗂️ **Elaborazione Batch:** Passa un singolo file o un'intera cartella per processare i file in sequenza.
* 🖼️ **Formati supportati:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.

---

## 🛠️ Requisiti e Installazione

Assicurati di avere Python 3 installato sul tuo sistema. Installa poi le dipendenze richieste tramite `pip`:

```bash
pip install torch torchvision pillow basicsr realesrgan
