"""TTS Server for VibeVox using Qwen3-TTS models."""

from __future__ import annotations

import io
import logging
import os
import uuid
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration — env-driven
MODEL_SIZE = os.environ.get("MODEL_SIZE", "1.7B")
DEFAULT_BASE_MODEL_ID = f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base"
DEFAULT_CUSTOM_VOICE_MODEL_ID = f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-CustomVoice"
DEFAULT_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_LANGUAGE = "English"

SUPPORTED_MODES = ["voice_clone", "custom_voice", "voice_design"]

CUSTOM_VOICE_SPEAKERS = {
    "Vivian": "Bright, slightly edgy young female voice — Chinese",
    "Serena": "Warm, gentle young female voice — Chinese",
    "Uncle_Fu": "Seasoned male voice with a low, mellow timbre — Chinese",
    "Dylan": "Youthful Beijing male voice with a clear, natural timbre — Chinese (Beijing Dialect)",
    "Eric": "Lively Chengdu male voice with a slightly husky brightness — Chinese (Sichuan Dialect)",
    "Ryan": "Dynamic male voice with strong rhythmic drive — English",
    "Aiden": "Sunny American male voice with a clear midrange — English",
    "Ono_Anna": "Playful Japanese female voice with a light, nimble timbre — Japanese",
    "Sohee": "Warm Korean female voice with rich emotion — Korean",
}


class TTSRequest(BaseModel):
    mode: str = "voice_design"
    text: str
    style_prompt: str = ""
    speaker: str = "Male_Narrator"
    language: str = DEFAULT_LANGUAGE
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    prompt_id: Optional[str] = None


class TTSServer:
    """Qwen3-TTS server with Triple-Model capabilities.

    Supports three synthesis modes:
      - voice_clone    → Base model       (generate_voice_clone)
      - custom_voice   → CustomVoice model (generate_custom_voice)
      - voice_design   → VoiceDesign model (generate_voice_design)
    """

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        custom_voice_model_id: str = DEFAULT_CUSTOM_VOICE_MODEL_ID,
        design_model_id: str = DEFAULT_DESIGN_MODEL_ID,
        device: str = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
        use_flash_attn: bool = True,
    ):
        self.base_model_id = base_model_id
        self.custom_voice_model_id = custom_voice_model_id
        self.design_model_id = design_model_id
        self.device = device
        self.dtype = dtype
        self.use_flash_attn = use_flash_attn
        self.model_base: Optional[Qwen3TTSModel] = None
        self.model_custom_voice: Optional[Qwen3TTSModel] = None
        self.model_design: Optional[Qwen3TTSModel] = None
        self._prompt_cache: dict[str, list] = {}

    def _load_single_model(self, model_id: str) -> Qwen3TTSModel:
        attn_implementation = "flash_attention_2" if self.use_flash_attn else "eager"
        try:
            logger.info(f"Loading model: {model_id} ...")
            return Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_implementation,
            )
        except Exception as e:
            logger.error(f"Failed to load {model_id} with flash_attn: {e}")
            if self.use_flash_attn:
                logger.info("Retrying with eager attention ...")
                return Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation="eager",
                )
            raise e

    def load_model(self):
        should_load_base = os.environ.get("LOAD_BASE_MODEL", "true").lower() == "true"
        should_load_custom = os.environ.get("LOAD_CUSTOM_VOICE_MODEL", "false").lower() == "true"
        should_load_design = os.environ.get("LOAD_DESIGN_MODEL", "true").lower() == "true"

        all_loaded = True
        if should_load_base and self.model_base is None:
            all_loaded = False
        if should_load_custom and self.model_custom_voice is None:
            all_loaded = False
        if should_load_design and self.model_design is None:
            all_loaded = False
        if all_loaded:
            logger.info("Requested models already loaded.")
            return

        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")

        try:
            if should_load_base and self.model_base is None:
                self.model_base = self._load_single_model(self.base_model_id)
                logger.info("✓ Base (Voice Clone) Model loaded!")

            if should_load_custom and self.model_custom_voice is None:
                self.model_custom_voice = self._load_single_model(self.custom_voice_model_id)
                logger.info("✓ CustomVoice Model loaded!")

            if should_load_design and self.model_design is None:
                self.model_design = self._load_single_model(self.design_model_id)
                logger.info("✓ VoiceDesign Model loaded!")
        except Exception as e:
            logger.error(f"Critical error loading models: {e}")
            raise

    # ── Prompt Caching -----------------------------------------------------

    def build_clone_prompt(self, reference_audio: str, reference_text: str | None = None) -> str:
        if self.model_base is None:
            raise RuntimeError("Base model not loaded — cannot build clone prompt")
        prompt_id = uuid.uuid4().hex
        logger.info("Building voice clone prompt …")
        prompt_items = self.model_base.create_voice_clone_prompt(
            ref_audio=reference_audio,
            ref_text=reference_text or "",
            x_vector_only_mode=False,
        )
        self._prompt_cache[prompt_id] = prompt_items
        logger.info(f"✓ Cached clone prompt: {prompt_id}")
        return prompt_id

    # ── Synthesis dispatch -------------------------------------------------

    def synthesize(
        self,
        text: str,
        mode: str = "voice_design",
        style_prompt: str = "",
        language: str = DEFAULT_LANGUAGE,
        speaker: str = "Male_Narrator",
        reference_audio: str | None = None,
        reference_text: str | None = None,
        prompt_id: str | None = None,
    ) -> tuple[bytes, int]:
        logger.info(f"Mode={mode}, text='{text[:60]}...'")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from {SUPPORTED_MODES}")

        try:
            if mode == "voice_clone":
                return self._synthesize_clone(text, language, reference_audio, reference_text, prompt_id)
            if mode == "custom_voice":
                return self._synthesize_custom_voice(text, language, speaker, style_prompt)
            return self._synthesize_design(text, language, style_prompt)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def _synthesize_clone(
        self,
        text: str,
        language: str,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        prompt_id: str | None = None,
    ) -> tuple[bytes, int]:
        if self.model_base is None:
            raise HTTPException(status_code=503, detail="Voice Clone model not loaded")

        if prompt_id and prompt_id in self._prompt_cache:
            logger.info("Using cached voice clone prompt")
            wavs, sr = self.model_base.generate_voice_clone(
                text=text, language=language, voice_clone_prompt=self._prompt_cache[prompt_id],
            )
        elif reference_audio:
            wavs, sr = self.model_base.generate_voice_clone(
                text=text, language=language, ref_audio=reference_audio, ref_text=reference_text or "",
            )
        else:
            raise ValueError("reference_audio or prompt_id required for voice_clone mode")

        return self._wav_bytes(wavs[0], sr)

    def _synthesize_custom_voice(
        self, text: str, language: str, speaker: str, instruct: str,
    ) -> tuple[bytes, int]:
        if self.model_custom_voice is None:
            raise HTTPException(status_code=503, detail="CustomVoice model not loaded")
        wavs, sr = self.model_custom_voice.generate_custom_voice(
            text=text, language=language, speaker=speaker, instruct=instruct or "",
        )
        return self._wav_bytes(wavs[0], sr)

    def _synthesize_design(
        self, text: str, language: str, instruct: str,
    ) -> tuple[bytes, int]:
        if self.model_design is None:
            raise HTTPException(status_code=503, detail="VoiceDesign model not loaded")
        if not instruct:
            raise ValueError("Style prompt (instruct) required for VoiceDesign mode")
        wavs, sr = self.model_design.generate_voice_design(
            text=text, language=language, instruct=instruct,
        )
        return self._wav_bytes(wavs[0], sr)

    @staticmethod
    def _wav_bytes(wav: torch.Tensor | list, sr: int) -> tuple[bytes, int]:
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        return buf.getvalue(), sr


# ── FastAPI app ------------------------------------------------------------

tts_server = TTSServer()
app = FastAPI(title="VibeVox TTS Server")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting VibeVox TTS Server …")
    tts_server.load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modes": SUPPORTED_MODES,
        "models_loaded": {
            "base": tts_server.model_base is not None,
            "custom_voice": tts_server.model_custom_voice is not None,
            "design": tts_server.model_design is not None,
        },
        "model_ids": {
            "base": tts_server.base_model_id,
            "custom_voice": tts_server.custom_voice_model_id,
            "design": tts_server.design_model_id,
        },
        "device": tts_server.device,
        "prompts_cached": len(tts_server._prompt_cache),
    }


@app.get("/speakers")
def list_speakers():
    return {"speakers": CUSTOM_VOICE_SPEAKERS}


@app.post("/tts/build_prompt")
def build_clone_prompt(request: TTSRequest):
    if not request.reference_audio:
        raise HTTPException(status_code=400, detail="reference_audio is required")
    try:
        prompt_id = tts_server.build_clone_prompt(
            reference_audio=request.reference_audio,
            reference_text=request.reference_text,
        )
        return {"prompt_id": prompt_id}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/tts")
def synthesize_speech(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        audio_bytes, sample_rate = tts_server.synthesize(
            text=request.text,
            mode=request.mode,
            style_prompt=request.style_prompt,
            language=request.language,
            speaker=request.speaker,
            reference_audio=request.reference_audio,
            reference_text=request.reference_text,
            prompt_id=request.prompt_id,
        )
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="output.wav"',
                "X-Sample-Rate": str(sample_rate),
            },
        )
    except Exception as e:
        logger.error(f"TTS request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "5000"))

    logger.info(f"Starting TTS server on {host}:{port}")
    uvicorn.run("tts_server:app", host=host, port=port, reload=False, log_level="info")
