import asyncio
import io
import os
import tempfile
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyrootutils
import soundfile as sf
from loguru import logger
from pydub import AudioSegment

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference
from tools.server.inference import PartialResultError

# 获取项目根目录
try:
    PROJECT_ROOT = Path(pyrootutils.find_root(search_from=__file__, indicator=".project-root"))
except Exception:
    # 如果找不到项目根目录，使用当前文件所在目录向上查找
    PROJECT_ROOT = Path(__file__).parent.parent.parent


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"  # 等待中
    RUNNING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class AsyncTask:
    """异步任务类"""
    
    def __init__(self, step_id: str, request: ServeTTSRequest):
        self.step_id = step_id
        self.request = request
        self.status = TaskStatus.PENDING
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error_message: Optional[str] = None
        self.result_path: Optional[str] = None
        self.cancelled = threading.Event()
        self.thread: Optional[threading.Thread] = None
        
    def cancel(self):
        """取消任务"""
        self.cancelled.set()
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.CANCELLED
        elif self.status == TaskStatus.RUNNING:
            self.status = TaskStatus.CANCELLED


class TaskManager:
    """任务管理器，用于管理异步生成任务"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        初始化任务管理器
        
        Args:
            temp_dir: 临时目录路径，用于存储生成结果。如果为None，则使用项目目录下的临时文件夹
        """
        self.tasks: Dict[str, AsyncTask] = {}
        self.lock = threading.Lock()
        
        # 设置临时目录
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            # 默认使用项目目录下的临时文件夹
            self.temp_dir = PROJECT_ROOT / "temp" / "async_results"
        
        # 确保临时目录存在
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TaskManager initialized with temp_dir: {self.temp_dir}")
    
    def create_task(self, step_id: str, request: ServeTTSRequest) -> AsyncTask:
        """
        创建新任务
        
        Args:
            step_id: 任务ID
            request: TTS请求
            
        Returns:
            AsyncTask对象
        """
        with self.lock:
            # 如果任务已存在
            if step_id in self.tasks:
                old_task = self.tasks[step_id]
                # 如果旧任务还在运行或等待中，抛出错误
                if old_task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    raise ValueError(f"Task with step_id {step_id} is running")
                # 如果任务已结束（已完成、失败或已取消），则覆盖
                # 清理旧任务的临时文件
                if old_task.result_path and os.path.exists(old_task.result_path):
                    try:
                        os.unlink(old_task.result_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete old task temp file: {e}")
                logger.info(f"Overwriting completed task {step_id}")
            
            task = AsyncTask(step_id, request)
            self.tasks[step_id] = task
            logger.info(f"Created task {step_id}")
            return task
    
    def get_task(self, step_id: str) -> Optional[AsyncTask]:
        """获取任务"""
        with self.lock:
            return self.tasks.get(step_id)
    
    def start_task(
        self, 
        task: AsyncTask, 
        engine: TTSInferenceEngine
    ) -> None:
        """
        启动任务执行
        
        Args:
            task: 异步任务
            engine: TTS推理引擎
        """
        def run_task():
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                # 检查是否已取消
                if task.cancelled.is_set():
                    task.status = TaskStatus.CANCELLED
                    return
                
                # 执行推理
                # 对于非流式请求，inference_wrapper会返回多个segment（bytes）和一个final（numpy数组）
                # 我们需要等待final结果，它包含了所有合并后的音频数据
                audio_data = None
                sample_rate = engine.decoder_model.sample_rate
                
                inference_error_occurred = False
                inference_error_message = None
                
                try:
                    for result in inference(task.request, engine):
                        # 检查是否已取消
                        if task.cancelled.is_set():
                            task.status = TaskStatus.CANCELLED
                            # 清理临时文件
                            if task.result_path and os.path.exists(task.result_path):
                                try:
                                    os.unlink(task.result_path)
                                except Exception as e:
                                    logger.warning(f"Failed to delete temp file: {e}")
                            return
                        
                        # 收集音频数据
                        # inference_wrapper对于segment返回bytes，对于final返回numpy数组
                        # 对于非流式请求，我们应该等待final结果（numpy数组），它包含了所有合并后的音频
                        if isinstance(result, np.ndarray):
                            # Final数据，直接使用（这是合并后的完整音频）
                            audio_data = result
                            # 收到 final 后，状态保持为 processing，继续处理可能的错误
                            # 不立即 break，让生成器自然结束或处理后续的 error
                        # 对于segment（bytes），我们忽略它们，因为final结果已经包含了所有数据
                except PartialResultError as partial_error:
                    # 部分结果错误：推理过程中发生错误，但已生成部分结果
                    inference_error_occurred = True
                    inference_error_message = partial_error.error_message
                    
                    # 如果已经收集到了final数据，仍然可以保存，状态保持为 processing
                    if audio_data is None:
                        # 没有收集到数据，重新抛出异常，状态会改为 failed
                        raise partial_error
                    # 有数据，记录警告但继续保存，状态保持为 processing
                    logger.warning(
                        f"Inference error occurred after final data was generated: {partial_error.error_message}. "
                        f"Will save partial result. Status will remain as processing until file is saved."
                    )
                except Exception as inference_error:
                    # 其他异常：如果推理过程中发生异常
                    inference_error_occurred = True
                    inference_error_message = str(inference_error)
                    
                    # 如果已经收集到了final数据，仍然可以保存，状态保持为 processing
                    if audio_data is None:
                        # 没有收集到数据，重新抛出异常，状态会改为 failed
                        raise inference_error
                    # 有数据，记录警告但继续保存，状态保持为 processing
                    logger.warning(
                        f"Inference error occurred after final data was generated: {inference_error}. "
                        f"Will save partial result. Status will remain as processing until file is saved."
                    )
                
                if audio_data is None:
                    raise ValueError("No audio generated")
                
                # 如果发生了错误但已有数据，状态仍然保持为 processing
                # 只有在保存文件完成后，状态才会改为 completed
                if inference_error_occurred:
                    logger.info(
                        f"Task {task.step_id} encountered error but has partial result. "
                        f"Status remains as processing. Will save file and then mark as completed."
                    )
                
                # 保存到临时文件
                result_filename = f"{task.step_id}.{task.request.format}"
                result_path = self.temp_dir / result_filename
                
                # 对于 MP3 格式，使用 pydub 设置比特率（至少 128kbps）
                if task.request.format == "mp3":
                    # 先将 numpy 数组转换为 AudioSegment
                    # 需要先保存为 WAV 格式的临时文件，然后转换为 MP3
                    temp_wav = io.BytesIO()
                    sf.write(temp_wav, audio_data, sample_rate, format="wav")
                    temp_wav.seek(0)
                    
                    # 使用 pydub 转换为 MP3，设置比特率为 192kbps
                    audio_segment = AudioSegment.from_wav(temp_wav)
                    audio_segment.export(
                        result_path,
                        format="mp3",
                        bitrate="192k"  # 设置为 192kbps，高于 128kbps 要求
                    )
                else:
                    # 对于其他格式（WAV、FLAC），使用 soundfile
                    buffer = io.BytesIO()
                    sf.write(
                        buffer,
                        audio_data,
                        sample_rate,
                        format=task.request.format,
                    )
                    
                    with open(result_path, "wb") as f:
                        f.write(buffer.getvalue())
                
                task.result_path = str(result_path)
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                
                # 如果之前发生了错误，记录错误信息（但不影响完成状态）
                if inference_error_occurred:
                    task.error_message = inference_error_message
                    logger.info(
                        f"Task {task.step_id} completed with partial result (error occurred: {inference_error_message}) "
                        f"in {task.completed_at - task.started_at:.2f}s"
                    )
                else:
                    logger.info(
                        f"Task {task.step_id} completed in "
                        f"{task.completed_at - task.started_at:.2f}s"
                    )
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.completed_at = time.time()
                logger.error(f"Task {task.step_id} failed: {e}", exc_info=True)
        
        task.thread = threading.Thread(target=run_task, daemon=True)
        task.thread.start()
    
    def cancel_task(self, step_id: str) -> bool:
        """
        取消任务
        
        Args:
            step_id: 任务ID
            
        Returns:
            是否成功取消
        """
        with self.lock:
            task = self.tasks.get(step_id)
            if task is None:
                return False
            
            task.cancel()
            logger.info(f"Task {step_id} cancelled")
            return True
    
    def cancel_all_tasks(self) -> int:
        """
        取消所有运行中的任务
        
        Returns:
            取消的任务数量
        """
        count = 0
        with self.lock:
            for task in self.tasks.values():
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    task.cancel()
                    count += 1
        
        logger.info(f"Cancelled {count} tasks")
        return count
    
    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """
        清理旧任务和临时文件
        
        Args:
            max_age_seconds: 最大保留时间（秒），默认1小时
        """
        current_time = time.time()
        cleaned = 0
        
        with self.lock:
            tasks_to_remove = []
            for step_id, task in self.tasks.items():
                age = current_time - task.created_at
                if age > max_age_seconds:
                    # 删除临时文件
                    if task.result_path and os.path.exists(task.result_path):
                        try:
                            os.unlink(task.result_path)
                        except Exception as e:
                            logger.warning(f"Failed to delete temp file {task.result_path}: {e}")
                    
                    tasks_to_remove.append(step_id)
                    cleaned += 1
            
            for step_id in tasks_to_remove:
                del self.tasks[step_id]
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old tasks")
    
    def get_task_info(self, step_id: str) -> Optional[dict]:
        """
        获取任务信息
        
        Args:
            step_id: 任务ID
            
        Returns:
            任务信息字典，如果任务不存在则返回None
        """
        task = self.get_task(step_id)
        if task is None:
            return None
        
        info = {
            "step_id": task.step_id,
            "status": task.status.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
        }
        
        if task.status == TaskStatus.COMPLETED and task.result_path:
            info["download_url"] = f"/download_result/{task.step_id}"
        
        if task.status == TaskStatus.FAILED:
            info["error"] = task.error_message
        
        return info

