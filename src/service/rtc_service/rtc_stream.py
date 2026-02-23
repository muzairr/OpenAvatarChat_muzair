import asyncio
import json
import time
import uuid
import weakref
from typing import Optional, Dict

import numpy as np
# noinspection PyPackageRequirements
from fastrtc import AsyncAudioVideoStreamHandler, AudioEmitType, VideoEmitType
from loguru import logger

from chat_engine.common.client_handler_base import ClientHandlerDelegate, ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from engine_utils.interval_counter import IntervalCounter

def _get_h264_encoder_info():
    """Get H.264 encoder info dynamically to avoid circular imports"""
    try:
        from handlers.client.rtc_client import client_handler_rtc
        return client_handler_rtc._selected_h264_encoder, client_handler_rtc._actual_h264_encoder
    except Exception:
        return "unknown", None


class RtcStream(AsyncAudioVideoStreamHandler):
    def __init__(self,
                 session_id: Optional[str],
                 expected_layout="mono",
                 input_sample_rate=16000,
                 output_sample_rate=24000,
                 output_frame_size=480,
                 fps=30,
                 stream_start_delay = 0.5,
                 ):
        super().__init__(
            expected_layout=expected_layout,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            fps=fps
        )
        self.client_handler_delegate: Optional[ClientHandlerDelegate] = None
        self.client_session_delegate: Optional[ClientSessionDelegate] = None

        self.weak_factory: Optional[weakref.ReferenceType[RtcStream]] = None

        self.session_id = session_id
        self.stream_start_delay = stream_start_delay

        self.chat_channel = None
        self.first_audio_emitted = False

        self.quit = asyncio.Event()
        self.last_frame_time = 0

        self.emit_counter = IntervalCounter("emit counter")

        self.start_time = None
        self.timestamp_base = self.input_sample_rate

        self.streams: Dict[str, RtcStream] = {}


    # copy is used as create_instance in fastrtc
    def copy(self, **kwargs) -> AsyncAudioVideoStreamHandler:
        try:
            if self.client_handler_delegate is None:
                raise Exception("ClientHandlerDelegate is not set.")
            session_id = kwargs.get("webrtc_id", None)
            if session_id is None:
                session_id = uuid.uuid4().hex
            
            # Log codec information
            selected_encoder, _ = _get_h264_encoder_info()
            logger.debug(f"[{session_id}] H.264 encoder: {selected_encoder}")
            
            new_stream = RtcStream(
                session_id,
                expected_layout=self.expected_layout,
                input_sample_rate=self.input_sample_rate,
                output_sample_rate=self.output_sample_rate,
                output_frame_size=self.output_frame_size,
                fps=self.fps,
                stream_start_delay=self.stream_start_delay,
            )
            new_stream.weak_factory = weakref.ref(self)
            new_session_delegate = self.client_handler_delegate.start_session(
                session_id=session_id,
                timestamp_base=self.input_sample_rate,
            )
            new_stream.client_session_delegate = new_session_delegate
            if session_id in self.streams:
                msg = f"Stream {session_id} already exists."
                raise RuntimeError(msg)
            self.streams[session_id] = new_stream
            return new_stream
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to create stream: {e}")
            raise

    async def emit(self) -> AudioEmitType:
        try:
            # if not self.args_set.is_set():
            # await self.wait_for_args()

            if not self.first_audio_emitted:
                self.client_session_delegate.clear_data()
                self.first_audio_emitted = True

            while not self.quit.is_set():
                # Try to get audio data with a timeout
                try:
                    chat_data = await asyncio.wait_for(
                        self.client_session_delegate.get_data(EngineChannelType.AUDIO), 
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # No audio data available - return silence to keep stream alive
                    silence_samples = self.output_frame_size
                    silence = np.zeros((1, silence_samples), dtype=np.float32)
                    return self.output_sample_rate, silence
                    
                if chat_data is None or chat_data.data is None:
                    # Return silence instead of continuing the loop
                    silence_samples = self.output_frame_size
                    silence = np.zeros((1, silence_samples), dtype=np.float32)
                    return self.output_sample_rate, silence
                    
                audio_array = chat_data.data.get_main_data()
                if audio_array is None:
                    # Return silence instead of continuing the loop
                    silence_samples = self.output_frame_size
                    silence = np.zeros((1, silence_samples), dtype=np.float32)
                    return self.output_sample_rate, silence
                    
                sample_num = audio_array.shape[-1]
                self.emit_counter.add_property("audio_emit", sample_num / self.output_sample_rate)
                return self.output_sample_rate, audio_array
        except Exception as e:
            logger.opt(exception=e).error("Error in emit: ")
            raise

    async def video_emit(self) -> VideoEmitType:
        try:
            if not self.first_audio_emitted:
                await asyncio.sleep(0.1)
            
            # Log actual encoder being used (for verification)
            selected_encoder, actual_encoder = _get_h264_encoder_info()
            if actual_encoder:
                logger.debug(f"[{self.session_id}] Using H.264 encoder: {actual_encoder}")
            
            self.emit_counter.add_property("video_emit")
            
            while not self.quit.is_set():
                get_data_start = time.perf_counter()
                
                # Try to get video data with a timeout
                try:
                    video_frame_data: ChatData = await asyncio.wait_for(
                        self.client_session_delegate.get_data(EngineChannelType.VIDEO),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # No video data available - return blank frame (black screen)
                    blank_frame = np.zeros((512, 512, 3), dtype=np.uint8)
                    return blank_frame
                
                get_data_wait_time = time.perf_counter() - get_data_start
                
                # Log slow data retrieval
                if get_data_wait_time > 0.05:
                    logger.debug(f"[{self.session_id}] Slow video data retrieval: {get_data_wait_time:.3f}s")
                
                if video_frame_data is None or video_frame_data.data is None:
                    # Return blank frame instead of continuing the loop
                    blank_frame = np.zeros((512, 512, 3), dtype=np.uint8)
                    return blank_frame
                
                frame_data = video_frame_data.data.get_main_data().squeeze()
                if frame_data is None:
                    # Return blank frame instead of continuing the loop
                    blank_frame = np.zeros((512, 512, 3), dtype=np.uint8)
                    return blank_frame
                
                return frame_data
        except Exception as e:
            logger.opt(exception=e).error("Error in video_emit")
            raise

    async def receive(self, frame: tuple[int, np.ndarray]):
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        _, array = frame
        self.client_session_delegate.put_data(
            EngineChannelType.AUDIO,
            array,
            timestamp,
            self.input_sample_rate,
        )

    async def video_receive(self, frame):
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        self.client_session_delegate.put_data(
            EngineChannelType.VIDEO,
            frame,
            timestamp,
            self.fps,
        )

    def set_channel(self, channel):
            super().set_channel(channel)
            self.chat_channel = channel
            logger.info(f"Data channel set up for session, starting chat history forwarder")
            
            async def process_chat_history():
                role = None
                chat_id = None
                logger.info("Chat history forwarder task started")
                try:
                    while not self.quit.is_set():
                        chat_data = await self.client_session_delegate.get_data(EngineChannelType.TEXT)
                        if chat_data is None or chat_data.data is None:
                            continue
                        logger.info(f"Got chat data: type={chat_data.type}, message={chat_data.data.get_main_data()}")
                        current_role = 'human' if chat_data.type == ChatDataType.HUMAN_TEXT else 'avatar'
                        chat_id = uuid.uuid4().hex if current_role != role else chat_id
                        role = current_role
                        message_json = json.dumps({'type': 'chat', 'message': chat_data.data.get_main_data(), 
                                                            'id': chat_id, 'role': current_role})
                        logger.info(f"Sending chat message to frontend: {message_json}")
                        self.chat_channel.send(message_json)
                        
                        # Check if this is the end of avatar response (for text-only mode)
                        if chat_data.type == ChatDataType.AVATAR_TEXT and chat_data.data.get_meta('avatar_text_end', False):
                            logger.info("Avatar response ended - sending avatar_end signal and re-enabling VAD")
                            # AI finished responding
                            self.client_session_delegate.shared_states.ai_is_responding = False
                            # Send avatar_end to frontend
                            self.chat_channel.send(json.dumps({'type': 'avatar_end'}))
                            # Re-enable VAD so user can speak again
                            self.client_session_delegate.shared_states.enable_vad = True
                            
                except Exception as e:
                    logger.opt(exception=e).error("Error in process_chat_history")
            asyncio.create_task(process_chat_history())
                
            @channel.on("message")
            def _(message):
                logger.info(f"Received message Custom: {message}")
                try:
                    message = json.loads(message)
                except Exception as e:
                    logger.info(e)
                    message = {}

                if self.client_session_delegate is None:
                    return
                timestamp = self.client_session_delegate.get_timestamp()
                if timestamp[0] / timestamp[1] < self.stream_start_delay:
                    return
                logger.info(f'on_chat_datachannel: {message}')
    
                if message['type'] == 'stop_chat':
                    logger.info("STOP_CHAT signal received - emitting INTERRUPT")
                    self.client_session_delegate.shared_states.ai_is_responding = False
                    self.client_session_delegate.emit_signal(
                        ChatSignal(
                            type=ChatSignalType.INTERRUPT,
                            source_type=ChatSignalSourceType.CLIENT,
                            source_name="rtc",
                        )
                    )
                    # Re-enable VAD so user can speak again immediately
                    self.client_session_delegate.shared_states.enable_vad = True
                    logger.info("VAD re-enabled after interrupt")
                elif message['type'] == 'chat':
                    channel.send(json.dumps({'type': 'avatar_end'}))
                    # Mark that AI is now responding (for barge-in detection)
                    self.client_session_delegate.shared_states.ai_is_responding = True
                    logger.info("AI started responding - barge-in detection enabled")
                    # Don't disable VAD - we need it active to detect barge-in!
                    # The VAD handler will manage enable/disable based on ai_is_responding flag
                    self.client_session_delegate.emit_signal(
                        ChatSignal(
                            # begin a new round of responding
                            type=ChatSignalType.BEGIN,
                            stream_type=ChatDataType.AVATAR_AUDIO,
                            source_type=ChatSignalSourceType.CLIENT,
                            source_name="rtc",
                        )
                    )
                    self.client_session_delegate.put_data(
                        EngineChannelType.TEXT,
                        message['data'],
                        loopback=True
                    )
                # else:

                # channel.send(json.dumps({"type": "chat", "unique_id": unique_id, "message": message}))
          
    async def on_chat_datachannel(self, message: Dict, channel):
        # {"type":"chat",id:"标识属于同一段话", "message":"Hello, world!"}
        # unique_id = uuid.uuid4().hex
        pass
    def shutdown(self):
        self.quit.set()
        factory = None
        if self.weak_factory is not None:
            factory = self.weak_factory()
        if factory is None:
            factory = self
        self.client_session_delegate = None
        if factory.client_handler_delegate is not None:
            factory.client_handler_delegate.stop_session(self.session_id)
