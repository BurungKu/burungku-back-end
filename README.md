# 🔧 Contoh Request Client (Python / Colab)

Berikut contoh lengkap cara memanggil API Backend BurungKu.

# 🐦 1. Request Prediksi Gambar Burung

Endpoint: POST /predict
Field: file
Header: x-api-key

## cUrl
```bash
curl -X POST "https://burung-ku-be-d45c373c2814.herokuapp.com/predict" \
  -H "x-api-key: " \
  -F "file=@/path/to/your/image.jpg"
```

## Python/Dari colab
```bash
import requests

url = "https://burung-ku-be-d45c373c2814.herokuapp.com/predict"
API_KEY = ""

img_path = "/content/02a9c226-fde1-4490-b5d3-614a054d0cfa.jpg"

with open(img_path, "rb") as f:
    files = {
        "file": ("bird.jpg", f, "image/jpeg")
    }
    headers = {
        "x-api-key": API_KEY
    }
    response = requests.post(url, files=files, headers=headers)

print("Status:", response.status_code)
print("Raw response:")
print(response.text)

```

## ✔ Contoh Output yang Diharapkan

```bash
{
  "success": true,
  "top1": {
    "species": "Calidris bairdii",
    "score": 92.9,
    "iucn": "LC"
  },
  "topk": [
    {"species": "Calidris melanotos", "score": 2.88, "iucn": "LC"},
    {"species": "Calidris minutilla", "score": 2.29, "iucn": "LC"},
    {"species": "Calidris pusilla", "score": 1.2, "iucn": "LC"}
  ]
}
```

# 🎙️ 2. Request Prediksi Suara Burung

Endpoint: POST /predict_audio
Field: audio_file
Optional: species (boleh dikosongkan)
Header: x-api-key

⚠️ Catatan:

Untuk file .mp3 gunakan MIME type: "audio/mpeg"

Untuk file .wav gunakan: "audio/wav"

## Python/Dari colab

```bash
import requests

url = "https://burung-ku-be-d45c373c2814.herokuapp.com/predict_audio"
API_KEY = ""
audio_path = "/content/600081.mp3"

with open(audio_path, "rb") as f:
    response = requests.post(
        url,
        headers={"x-api-key": API_KEY},
        files={
            "audio_file": ("audio.mp3", f, "audio/mpeg")
        },
        data={
            "species": "Aix sponsa"  
        }
    )

print("Status:", response.status_code)
print("Raw response:")
print(response.text)
```
## cUrl

```bash

curl -X POST "https://burung-ku-be-d45c373c2814.herokuapp.com/predict_audio" \
  -H "x-api-key: AntekAsing" \
  -F "audio_file=@/path/to/audio.mp3" \
  -F "species=Aix sponsa"
```

# ✔ Contoh Output yang Diharapkan
{
  "success": true,
  "species": "Aix sponsa",
  "predicted_simple_type": "contact_call"
}