# Chatterbox TTS API (es-ES)

Partiendo del repo original https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-es

API HTTP basada en FastAPI para sintetizar voz en español con Chatterbox. Reemplaza la interfaz de Gradio y devuelve audio WAV directamente.

## Requisitos

- Docker y NVIDIA Container Toolkit.
- Driver NVIDIA compatible con la imagen `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`. Si el driver es antiguo, cambia la base y las ruedas de PyTorch a CUDA 12.6.
- GPU recomendada: 8 GB de VRAM o más.
- El directorio `models/` de la raíz debe existir; ahí se reutiliza la caché de HuggingFace.
- Las voces reutilizables se colocan en `voices/` (montado en `/app/voices` dentro del contenedor).
- `Dockerfile` y `docker-compose.yml` están en la raíz del proyecto.

## Arranque

Desde la raíz del proyecto:

```bash
docker compose up -d --build
```

La API queda disponible en `http://localhost:8020` y la documentación interactiva en `http://localhost:8020/docs`.

En un equipo sin GPU, define `TTS_DEVICE=cpu` y elimina `gpus: all` del compose para pruebas.

## Endpoints

### `GET /health`

Devuelve el estado del modelo, dispositivo y uso de VRAM.

```bash
curl http://localhost:8020/health
```

### `GET /voices`

Lista la voz incorporada y los audios de referencia disponibles en `voices/`.

```bash
curl http://localhost:8020/voices
```

Respuesta de ejemplo:

```json
{
  "default": "default",
  "count": 2,
  "voices": [
    {"id": "default", "name": "Voz incorporada", "type": "builtin", "language": "es"},
    {"id": "es_es_f1", "name": "es_es_f1", "type": "file", "filename": "es_es_f1.wav", "language": "es"}
  ]
}
```

Los archivos se detectan sin reiniciar la API. Formatos aceptados: `.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, `.m4a`.

### `POST /voices`

Sube un nuevo audio de referencia a `voices/`. Queda disponible inmediatamente en `GET /voices` y para `POST /tts`.

| Campo | Tipo | Obligatorio | Valor por defecto |
| --- | --- | --- | --- |
| `audio` | archivo | Sí | — |
| `voice_id` | texto | No | nombre del archivo |
| `overwrite` | bool | No | `false` |

```bash
curl -X POST http://localhost:8020/voices \
  -F "audio=@mi_voz.wav" \
  -F "voice_id=mi_voz"
```

Si la voz ya existe responde `409`; usa `-F "overwrite=true"` para reemplazarla. El ID `default` está reservado.

### `POST /tts`

Recibe `multipart/form-data` y devuelve `audio/wav`.

| Campo | Tipo | Obligatorio | Valor por defecto |
| --- | --- | --- | --- |
| `text` | texto | Sí | — |
| `reference_audio` | archivo | No | voz incorporada (`conds.pt`) |
| `voice` | texto | No | `default` (ID de `GET /voices`) |
| `exaggeration` | float | No | `0.5` |
| `temperature` | float | No | `0.8` |
| `cfg_weight` | float | No | `0.5` |
| `seed` | int | No | `0` (aleatorio) |
| `language_id` | texto | No | `es` |

Formatos de referencia aceptados: `.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, `.m4a`. Si se envía `reference_audio`, tiene prioridad sobre `voice`.

#### Solo texto

```bash
curl -X POST http://localhost:8020/tts \
  -F "text=Hola, ¿cómo estás? Hoy es un día perfecto para dar un paseo." \
  -o salida.wav
```

#### Con una voz de `voices/`

```bash
curl -X POST http://localhost:8020/tts \
  -F "text=El museo del Prado alberga obras de Velázquez y Goya." \
  -F "voice=es_es_f1" \
  -o salida.wav
```

#### Con clonación subiendo un archivo

```bash
curl -X POST http://localhost:8020/tts \
  -F "text=El museo del Prado alberga obras de Velázquez y Goya." \
  -F "reference_audio=@referencia.wav" \
  -F "exaggeration=0.5" \
  -F "temperature=0.8" \
  -F "cfg_weight=0.5" \
  -o salida.wav
```

## Variables de entorno

| Variable | Valor por defecto | Descripción |
| --- | --- | --- |
| `TTS_DEVICE` | `auto` | `auto`, `cuda` o `cpu`. |
| `TTS_WARMUP` | `1` | Calentamiento CUDA al iniciar. |
| `HF_TOKEN` | vacío | Token de HuggingFace si el modelo no está cacheado. |
| `HF_HOME` | `/models/huggingface` | Caché de modelos. |
| `VOICES_DIR` | `/app/voices` | Carpeta con audios de referencia reutilizables. |
| `MAX_UPLOAD_MB` | `25` | Tamaño máximo del audio de referencia. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce la fragmentación de VRAM. |

## Caché local y modo offline

El modelo intenta cargarse primero desde `models/huggingface` con `local_files_only=True`. Si la caché está completa, la API arranca sin conexión a HuggingFace. Solo si falta algún archivo se intenta la descarga remota.

## Optimizaciones GPU

- Modelo único residente en GPU y un solo worker de Uvicorn.
- Inferencia serializada con lock para evitar carreras en GPU.
- `float32` con TF32 habilitado (`matmul` y `cudnn`).
- Warmup opcional al arrancar para reducir la latencia de la primera petición.
- `/health` informa VRAM total, asignada y reservada.
