import re
from typing import Dict, Optional, cast
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field
from abc import ABC
import os

from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.contexts.session_context import SessionContext

from engine_utils.directory_info import DirectoryInfo
from engine_utils.general_slicer import SliceContext, slice_data

from faster_whisper import WhisperModel


class WhisperASRConfig(HandlerBaseConfigModel, BaseModel):
    model_size: str = Field(default="base")  # tiny, base, small, medium, large
    device: str = Field(default="auto")  # auto, cpu, cuda
    compute_type: str = Field(default="int8")  # int8, float16, float32
    language: str = Field(default="en")  # Force English
    beam_size: int = Field(default=5)
    vad_filter: bool = Field(default=True)  # Use Whisper's built-in VAD
    min_silence_duration_ms: int = Field(default=500)  # Minimum silence to split


class WhisperASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )
        self.cache = {}
        self.shared_states = None


class HandlerWhisperASR(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_size = 'base'
        self.device = 'auto'
        self.compute_type = 'int8'
        self.language = 'en'
        self.beam_size = 5
        self.vad_filter = True
        self.min_silence_duration_ms = 500

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="WhisperASR",
            config_model=WhisperASRConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, WhisperASRConfig):
            self.model_size = handler_config.model_size
            self.device = handler_config.device
            self.compute_type = handler_config.compute_type
            self.language = handler_config.language
            self.beam_size = handler_config.beam_size
            self.vad_filter = handler_config.vad_filter
            self.min_silence_duration_ms = handler_config.min_silence_duration_ms

        # Auto-detect device
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                self.device = "cpu"

        # Adjust compute_type based on device
        if self.device == "cpu":
            self.compute_type = "int8"  # CPU works best with int8
        elif self.compute_type == "float16" and self.device == "cpu":
            self.compute_type = "float32"  # CPU doesn't support float16

        logger.info(f"Loading Whisper model: {self.model_size} on {self.device} with {self.compute_type}")
        
        # Download and load model
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=os.path.join(DirectoryInfo.get_models_dir(), "whisper")
        )
        
        logger.info(f"Whisper model {self.model_size} loaded successfully")

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, WhisperASRConfig):
            handler_config = WhisperASRConfig()
        context = WhisperASRContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        return context
    
    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):

        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(WhisperASRContext, context)
        
        if inputs.type == ChatDataType.HUMAN_AUDIO:
            audio = inputs.data.get_main_data()
        else:
            return
            
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if audio is not None:
            audio = audio.squeeze()
            
            logger.debug(f'Audio chunk received: {audio.shape}')
            for audio_segment in slice_data(context.audio_slice_context, audio):
                if audio_segment is None or audio_segment.shape[0] == 0:
                    continue
                context.output_audios.append(audio_segment)

        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        # Collect all audio
        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate([
                    remainder_audio,
                    np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))
                ])
                context.output_audios.append(remainder_audio)
                
        if len(context.output_audios) == 0:
            logger.warning("No audio to process")
            context.shared_states.enable_vad = True
            return
            
        output_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()
        
        # Convert to float32 as required by faster-whisper
        audio_float = output_audio.astype(np.float32)
        
        # Normalize audio to [-1, 1] range
        max_val = np.abs(audio_float).max()
        if max_val > 0:
            audio_float = audio_float / max_val
        
        logger.info(f'Processing audio: duration={len(audio_float)/16000:.2f}s')
        
        try:
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_float,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters={
                    "min_silence_duration_ms": self.min_silence_duration_ms,
                    "threshold": 0.5  # Higher threshold = less sensitive to noise
                },
                condition_on_previous_text=False,  # Don't use context (prevents hallucination)
                temperature=0.0,  # Greedy decoding (more deterministic, less hallucination)
                compression_ratio_threshold=2.4,  # Reject if too repetitive
                log_prob_threshold=-1.0,  # Reject low-confidence outputs
                no_speech_threshold=0.6,  # Higher = more strict about detecting actual speech
            )
            
            # Collect all segments
            output_text = ""
            for segment in segments:
                output_text += segment.text + " "
            
            output_text = output_text.strip()
            
            logger.info(f"Whisper transcription: '{output_text}' (language: {info.language}, probability: {info.language_probability:.2f})")
            
            # Filter out garbage
            if len(output_text) == 0:
                logger.info("Empty transcription, re-enabling VAD")
                context.shared_states.enable_vad = True
                return
            
            # Check if it's actually English and makes sense
            if info.language != self.language:
                logger.warning(f"Detected language {info.language} instead of {self.language}, ignoring")
                context.shared_states.enable_vad = True
                return
            
            # Filter non-ASCII characters for English
            if self.language == "en":
                clean_text = re.sub(r'[^\x00-\x7F]+', '', output_text)
                clean_text = clean_text.strip()
                
                if len(clean_text) < len(output_text) * 0.5:
                    logger.warning(f"Too many non-English characters, ignoring: {output_text[:100]}")
                    context.shared_states.enable_vad = True
                    return
                    
                output_text = clean_text
            
            # Minimum length check (avoid single-letter hallucinations)
            if len(output_text) < 3:
                logger.warning(f"Transcription too short, likely noise: '{output_text}'")
                context.shared_states.enable_vad = True
                return
            
            # Output the transcription
            output = DataBundle(output_definition)
            output.set_main_data(output_text)
            output.add_meta('human_text_end', False)
            output.add_meta('speech_id', speech_id)
            yield output

            end_output = DataBundle(output_definition)
            end_output.set_main_data('')
            end_output.add_meta("human_text_end", True)
            end_output.add_meta("speech_id", speech_id)
            yield end_output
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            context.shared_states.enable_vad = True
            return

    def destroy_context(self, context: HandlerContext):
        pass
