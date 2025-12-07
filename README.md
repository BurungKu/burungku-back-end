# 🐦 BurungKu  
**Identifikasi Spesies Burung & Analisis Kesehatan dari Kicauan — Powered by AI**

BurungKu adalah aplikasi kecerdasan buatan yang dirancang untuk membantu pecinta burung, peneliti, hingga masyarakat umum dalam **mengidentifikasi spesies burung dari gambar**, serta **mendeteksi kondisi burung dari kicauannya**.  
Aplikasi ini juga memiliki visi jangka panjang sebagai platform konservasi dan ekosistem digital yang etis.

---

## 🚀 Fitur Utama

### 🔍 1. Deteksi Spesies dari Gambar  
- Model vision AI dilatih menggunakan dataset **iNatLoc**.  
- Mampu mengenali **106 spesies burung** lokal.  
- Mendukung input gambar dari kamera maupun upload file.  
- Output mencakup:
  - Nama spesies  
  - Skor probabilitas  
  - Informasi konservasi dasar

---

### 🎙️ 2. Analisis Kesehatan dari Kicauan  
- Model audio dilatih menggunakan ribuan rekaman dari **Xeno-Canto**.  
- Mengklasifikasi jenis kicauan untuk mendeteksi indikasi kondisi burung.  
- Didesain untuk analisis awal, bukan diagnosis medis.

---

## 🌱 Visi Jangka Panjang

### 🛒 Marketplace Satwa yang Etis  
Marketplace yang mempromosikan perdagangan legal dan bertanggung jawab.

### 🪪 Paspor Burung Digital  
Identitas digital burung untuk riwayat kesehatan, silsilah, dan kepemilikan.

### 📢 Platform Aduan Masyarakat  
Fitur untuk melaporkan perburuan liar, penyelundupan, atau burung terluka.

---

## 🧠 Model dan Dataset

### 1. **Image Bird Classifier**
- Dataset: **iNatLoc**  
- Jumlah spesies: **106 spesies**

### 2. **Bird Sound Condition Classifier**
- Dataset: **Xeno-Canto**  
- Preprocessing: Mel-spectrogram, normalisasi, segmentasi audio

---

## 📚 Sitasi Dataset

### **iNatLoc Dataset Citation**
Silakan gunakan sitasi resmi berikut jika menyebut dataset di makalah atau dokumentasi model:

**iNatLoc: A Benchmark for Fine-Grained Local Bird Classification**  
*Vincent, Hugo; Ben-Younes, Hedi; Russell, Bryan C.; Jégou, Hervé.*  
2023.  
Repository: https://github.com/HugoTian/iNatLoc  
(Atau sesuai format sitasi akademik)

```bibtex
@misc{inatloc2023,
  title        = {iNatLoc: A Benchmark for Fine-Grained Local Bird Classification},
  author       = {Vincent, Hugo and Ben-Younes, Hedi and Russell, Bryan C. and Jégou, Hervé},
  year         = {2023},
  url          = {https://github.com/HugoTian/iNatLoc},
}

@misc{xenocanto,
  title        = {Xeno-Canto: Sharing bird sounds from around the world},
  author       = {Xeno-Canto Foundation},
  url          = {https://www.xeno-canto.org}
}
