import io
import logging
import os
import random
import re
import tempfile
import threading
import unicodedata
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from chatterbox.tts import ChatterboxTTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("chatterbox-api")

MODEL: Optional[ChatterboxTTS] = None
DEFAULT_CONDS = None
DEVICE = "cpu"
MODEL_LOCK = threading.Lock()

FIXED_LANGUAGE_ID = "es"
MAX_CHARS = 300
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
BUILTIN_VOICE_ID = "default"
BUILTIN_VOICE_NAME = "Voz incorporada"
VOICES_DIR = Path(os.getenv("VOICES_DIR", str(Path(__file__).resolve().parent / "voices")))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_MB", 25) * 1024 * 1024
WARMUP_ENABLED = _env_flag("TTS_WARMUP", default=True)


def resolve_device() -> str:
    requested = os.getenv("TTS_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        logger.warning("Valor TTS_DEVICE=%r no válido; usando 'auto'", requested)
        requested = "auto"

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("TTS_DEVICE=cuda solicitado pero CUDA no está disponible; usando CPU")
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def configure_gpu(device: str) -> None:
    if device != "cuda":
        logger.info("Ejecutando en CPU (sin optimizaciones CUDA)")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    try:
        props = torch.cuda.get_device_properties(0)
        total_gib = props.total_memory / (1024 ** 3)
        logger.info(
            "CUDA lista: %s | %.1f GiB | compute capability %d.%d",
            props.name,
            total_gib,
            props.major,
            props.minor,
        )
    except Exception:
        logger.exception("No se pudo consultar la información de la GPU")


def cuda_info() -> dict:
    if not torch.cuda.is_available():
        return {}
    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu": props.name,
            "vram_total_gb": round(props.total_memory / (1024 ** 3), 2),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
        }
    except Exception:
        return {}


def list_available_voices() -> list[dict]:
    voices = [
        {
            "id": BUILTIN_VOICE_ID,
            "name": BUILTIN_VOICE_NAME,
            "type": "builtin",
            "language": FIXED_LANGUAGE_ID,
        }
    ]

    used_ids = {BUILTIN_VOICE_ID}
    try:
        entries = sorted(VOICES_DIR.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        entries = []

    for path in entries:
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            continue

        base = path.stem.strip() or path.name
        voice_id = base
        counter = 2
        while voice_id in used_ids:
            voice_id = f"{base}-{counter}"
            counter += 1
        used_ids.add(voice_id)

        voices.append(
            {
                "id": voice_id,
                "name": base,
                "type": "file",
                "filename": path.name,
                "language": FIXED_LANGUAGE_ID,
            }
        )

    return voices


def resolve_voice_path(voice_id: str) -> Optional[Path]:
    voice_id = (voice_id or BUILTIN_VOICE_ID).strip()
    if not voice_id or voice_id == BUILTIN_VOICE_ID:
        return None

    for voice in list_available_voices():
        if voice["id"] == voice_id:
            return VOICES_DIR / voice["filename"]

    raise HTTPException(
        status_code=404,
        detail=f"Voz no encontrada: {voice_id}. Consulta GET /voices",
    )


def safe_voice_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_value).strip("._-")
    return safe[:64] or "voice"


def save_voice_upload(
    audio: UploadFile,
    voice_id: Optional[str],
    overwrite: bool,
) -> dict:
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        formats = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise HTTPException(
            status_code=400,
            detail=f"Formato de audio no soportado ({suffix or 'sin extensión'}). Usa: {formats}",
        )

    data = audio.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"El audio supera {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    if not data:
        raise HTTPException(status_code=400, detail="El archivo de audio está vacío")

    stem_source = (voice_id or Path(audio.filename or "").stem or "voice").strip()
    if Path(stem_source).suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
        stem_source = Path(stem_source).stem

    stem = safe_voice_stem(stem_source)
    if stem == BUILTIN_VOICE_ID:
        raise HTTPException(
            status_code=400,
            detail="El ID 'default' está reservado para la voz incorporada",
        )

    destination = VOICES_DIR / f"{stem}{suffix}"
    try:
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo crear la carpeta de voces: {exc}",
        ) from exc

    if destination.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Ya existe {destination.name}; usa overwrite=true para reemplazarlo",
        )

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=VOICES_DIR, delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = Path(tmp.name)
        os.replace(temp_path, destination)
        temp_path = None
        os.chmod(destination, 0o644)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    for voice in list_available_voices():
        if voice.get("filename") == destination.name:
            return voice

    return {
        "id": stem,
        "name": stem,
        "type": "file",
        "filename": destination.name,
        "language": FIXED_LANGUAGE_ID,
    }


def load_model(device: str) -> ChatterboxTTS:
    logger.info("Cargando modelo Chatterbox TTS en %s", device)
    try:
        model = ChatterboxTTS.from_pretrained(device)
    except Exception:
        if device != "cuda":
            raise
        logger.exception("Falló la inicialización en CUDA; se reintenta en CPU")
        model = ChatterboxTTS.from_pretrained("cpu")
    logger.info("Modelo cargado en %s", model.device)
    return model


def set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def warmup_model() -> None:
    if not WARMUP_ENABLED:
        return
    if MODEL is None or MODEL.device != "cuda":
        return
    if DEFAULT_CONDS is None:
        logger.warning("Sin voz incorporada; se omite el warmup")
        return

    logger.info("Ejecutando warmup de CUDA")
    try:
        with MODEL_LOCK:
            MODEL.conds = DEFAULT_CONDS
            MODEL.generate("Hola.", language_id=FIXED_LANGUAGE_ID)
        torch.cuda.synchronize()
        logger.info("Warmup completado")
    except Exception:
        logger.exception("El warmup falló; la API seguirá disponible")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, DEFAULT_CONDS, DEVICE
    DEVICE = resolve_device()
    configure_gpu(DEVICE)
    MODEL = load_model(DEVICE)
    DEVICE = MODEL.device
    DEFAULT_CONDS = MODEL.conds
    if DEFAULT_CONDS is None:
        logger.warning("El modelo no tiene voz incorporada; reference_audio será obligatorio")
    warmup_model()

    try:
        yield
    finally:
        MODEL = None
        DEFAULT_CONDS = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="Chatterbox TTS API",
    description="API de síntesis de voz en español (es-ES) basada en Chatterbox.",
    version="1.0.0",
    lifespan=lifespan,
)


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    samples = np.asarray(samples, dtype=np.float32).squeeze()
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def synthesize(
    text: str,
    reference_path: Optional[Path],
    exaggeration: float,
    temperature: float,
    cfg_weight: float,
    seed: int,
    language_id: str,
) -> tuple[np.ndarray, int]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado")
    if reference_path is None and DEFAULT_CONDS is None:
        raise HTTPException(
            status_code=503,
            detail="No hay voz incorporada; envía reference_audio para generar",
        )

    lang = (language_id or FIXED_LANGUAGE_ID).strip().lower()

    with MODEL_LOCK:
        if seed != 0:
            set_seed(seed, MODEL.device)

        MODEL.conds = DEFAULT_CONDS
        generate_kwargs = {
            "text": text,
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
            "language_id": lang,
        }
        if reference_path is not None:
            generate_kwargs["audio_prompt_path"] = str(reference_path)

        wav = MODEL.generate(**generate_kwargs)
        sample_rate = MODEL.sr

    return wav.squeeze(0).detach().cpu().numpy(), sample_rate


@app.get("/", tags=["info"])
def root() -> dict:
    return {
        "name": "Chatterbox TTS API",
        "language": FIXED_LANGUAGE_ID,
        "docs": "/docs",
        "health": "/health",
        "voices": "GET /voices",
        "upload_voice": "POST /voices",
        "tts": "POST /tts",
    }


@app.get("/voices", tags=["voices"])
def voices() -> dict:
    available = list_available_voices()
    return {
        "default": BUILTIN_VOICE_ID,
        "count": len(available),
        "voices": available,
    }


@app.post("/voices", tags=["voices"], status_code=201)
def upload_voice(
    audio: UploadFile = File(..., description="Archivo de audio de referencia"),
    voice_id: Optional[str] = Form(
        default=None,
        description="ID/nombre de la voz; si se omite, se usa el nombre del archivo",
    ),
    overwrite: bool = Form(
        default=False,
        description="Reemplazar la voz si ya existe",
    ),
) -> dict:
    voice = save_voice_upload(audio, voice_id, overwrite)
    return {
        "message": "Voz subida correctamente",
        "voice": voice,
    }


@app.get("/health", tags=["info"])
def health() -> dict:
    payload = {
        "status": "ok" if MODEL is not None else "unavailable",
        "device": MODEL.device if MODEL is not None else DEVICE,
        "model_loaded": MODEL is not None,
        "cuda_available": torch.cuda.is_available(),
    }
    payload.update(cuda_info())
    return payload


@app.post(
    "/tts",
    tags=["tts"],
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
def text_to_speech(
    text: str = Form(..., description=f"Texto a sintetizar (máx. {MAX_CHARS} caracteres)"),
    reference_audio: Optional[UploadFile] = File(
        default=None,
        description="Audio de referencia opcional para clonar la voz",
    ),
    voice: str = Form(
        BUILTIN_VOICE_ID,
        description="ID de voz de GET /voices; 'default' usa la voz incorporada",
    ),
    exaggeration: float = Form(0.5, ge=0.25, le=2.0),
    temperature: float = Form(0.8, ge=0.05, le=5.0),
    cfg_weight: float = Form(0.5, ge=0.2, le=1.0),
    seed: int = Form(0, ge=0),
    language_id: str = Form(FIXED_LANGUAGE_ID),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")
    if len(text) > MAX_CHARS:
        logger.info("Texto recortado de %d a %d caracteres", len(text), MAX_CHARS)
        text = text[:MAX_CHARS]

    temp_path: Optional[Path] = None
    voice_path: Optional[Path] = None
    try:
        if reference_audio is not None:
            suffix = Path(reference_audio.filename or "").suffix.lower()
            if suffix not in SUPPORTED_AUDIO_SUFFIXES:
                formats = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
                raise HTTPException(
                    status_code=400,
                    detail=f"Formato de audio no soportado ({suffix or 'sin extensión'}). Usa: {formats}",
                )

            data = reference_audio.file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"El audio de referencia supera {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            if not data:
                raise HTTPException(status_code=400, detail="El audio de referencia está vacío")

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                temp_path = Path(tmp.name)
        else:
            voice_path = resolve_voice_path(voice)

        samples, sample_rate = synthesize(
            text=text,
            reference_path=temp_path or voice_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            seed=seed,
            language_id=language_id,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return Response(
        content=wav_bytes(samples, sample_rate),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="speech.wav"',
            "X-Sample-Rate": str(sample_rate),
        },
    )
