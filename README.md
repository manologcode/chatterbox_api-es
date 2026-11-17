# Chatterbox TTS API (es-ES)

Partiendo del repo original de

API HTTP para convertir texto en voz en español usando el modelo **Chatterbox Multilingual es-ES**. Sustituye la antigua interfaz de Gradio y devuelve audio WAV directamente.

## Qué incluye

- `api/`: aplicación FastAPI y código del modelo.
- `voices/`: audios de referencia reutilizables, montados en `/app/voices`.
- `models/`: caché de HuggingFace reutilizada por el contenedor.
- `Dockerfile` y `docker-compose.yml`: despliegue con GPU NVIDIA.

## Cómo funciona

1. Al arrancar, se carga una única instancia del modelo en la GPU (o CPU si no hay CUDA). Primero se intenta usar la caché local de `models/`; si está completa, no se necesita conexión a HuggingFace.
2. `POST /tts` recibe el texto y parámetros de generación.
3. El modelo genera la onda de audio y la API la devuelve como `audio/wav`.
4. Se puede usar la voz incorporada, una voz guardada en `voices/` o un audio de referencia subido en la petición.

La inferencia está serializada para evitar problemas de concurrencia en GPU. Se usa `float32` con TF32 y un warmup opcional para reducir la latencia de la primera petición.

## Arranque rápido (desde el repositorio)

```bash
docker compose up -d --build
```

Si vienes de la versión anterior con Gradio, elimina primero el contenedor antiguo:

```bash
docker compose down --remove-orphans
docker rm -f chatterbox-es
docker compose up -d --build
```

La API queda en `http://localhost:8020` y la documentación interactiva en `http://localhost:8020/docs`.

## Usar la imagen publicada sin clonar el repositorio

La imagen `manologcode/chatterbox_api-es:latest` se publica en Docker Hub. No hace falta clonar el repositorio: crea un `docker-compose.yml` con este contenido y arráncalo. Los modelos y las voces se guardan en **volúmenes con nombre**, de modo que se descargan/suben una sola vez y se reutilizan al actualizar la imagen.

```yaml
services:
  chatterbox-es-api:
    image: manologcode/chatterbox_api-es:latest
    container_name: chatterbox-es-api
    restart: unless-stopped
    ports:
      - "8020:8000"
    gpus: all
    shm_size: "2gb"
    environment:
      - HF_HOME=/models/huggingface
      - VOICES_DIR=/app/voices
      - TTS_DEVICE=auto
      - TTS_WARMUP=1
      - HF_TOKEN=${HF_TOKEN:-}
    volumes:
      - chatterbox-models:/models
      - chatterbox-voices:/app/voices

volumes:
  chatterbox-models:
  chatterbox-voices:
```

```bash
docker compose up -d
```

- `chatterbox-models` guarda la caché de HuggingFace. La primera vez se descarga el modelo; en arranques posteriores se reutiliza sin volver a descargar.
- `chatterbox-voices` guarda las voces subidas con `POST /voices`. Sobreviven a reinicios y actualizaciones.

Actualizar a una nueva versión conservando modelos y voces:

```bash
docker compose pull
docker compose up -d
```

Si prefieres ver los archivos en tu disco en lugar de usar volúmenes con nombre, cambia los volúmenes del servicio y elimina la sección `volumes:` final del compose:

```yaml
    volumes:
      - ./models:/models
      - ./voices:/app/voices
```

```bash
mkdir -p models voices
```

La inferencia con GPU NVIDIA requiere un host `linux/amd64` con el NVIDIA Container Toolkit instalado.

Generar audio:

```bash
curl -X POST http://localhost:8020/tts \
  -F "text=Hola, ¿cómo estás? Esto es una prueba de voz." \
  -o salida.wav
```

## Publicar la imagen en Docker Hub

El workflow de GitHub Actions [`.github/workflows/image.yml`](.github/workflows/image.yml) construye y publica la imagen en cada push a `main`, en tags `v*` y manualmente con `workflow_dispatch`.

Configura en el repositorio de GitHub los secretos:

| Secreto | Descripción |
| --- | --- |
| `DOCKER_USERNAME` | Usuario de Docker Hub (`manologcode`). |
| `DOCKER_PASSWORD` | Token de acceso de Docker Hub. |

Para publicar manualmente desde tu equipo:

```bash
docker login
docker build -t manologcode/chatterbox_api-es:latest .
docker push manologcode/chatterbox_api-es:latest
```

## Endpoints principales

| Método | Ruta | Descripción |
| --- | --- | --- |
| `GET` | `/health` | Estado del modelo, GPU y VRAM. |
| `GET` | `/voices` | Lista las voces disponibles. |
| `POST` | `/voices` | Sube un nuevo audio de referencia. |
| `POST` | `/tts` | Genera y devuelve el audio WAV. |

## Estructura

```text
.
├── api/
│   ├── app.py
│   ├── chatterbox/
│   ├── requirements.txt
│   └── README.md
├── voices/
├── models/
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

La documentación detallada de la API está en [`api/README.md`](api/README.md).

## Requisitos

- Docker y NVIDIA Container Toolkit.
- GPU NVIDIA con 8 GB de VRAM o más.
- Driver compatible con CUDA 12.8. Si el driver es anterior, cambia la imagen base y las ruedas de PyTorch a CUDA 12.6.

## Variables de entorno

| Variable | Valor por defecto | Descripción |
| --- | --- | --- |
| `TTS_DEVICE` | `auto` | `auto`, `cuda` o `cpu`. |
| `TTS_WARMUP` | `1` | Calentamiento de CUDA al iniciar. |
| `HF_TOKEN` | vacío | Token de HuggingFace si el modelo no está cacheado. |
| `HF_HOME` | `/models/huggingface` | Caché de modelos. |
| `VOICES_DIR` | `/app/voices` | Carpeta de voces dentro del contenedor. |
| `MAX_UPLOAD_MB` | `25` | Tamaño máximo de audio de referencia. |
