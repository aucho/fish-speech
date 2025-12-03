import io
import os
import re
import shutil
import tempfile
import time
from http import HTTPStatus
from pathlib import Path

import numpy as np
import ormsgpack
import soundfile as sf
import torch
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    StreamResponse,
    UploadFile,
    request,
)
from loguru import logger
from typing_extensions import Annotated

from fish_speech.utils.schema import (
    AddReferenceRequest,
    AddReferenceResponse,
    AsyncGenerateRequest,
    AsyncGenerateResponse,
    DeleteReferenceResponse,
    ListReferencesResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    StopTaskResponse,
    TaskStatusResponse,
    UpdateReferenceResponse,
)
from tools.server.api_utils import (
    buffer_to_async_generator,
    format_response,
    get_content_type,
    inference_async,
)
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)
from tools.server.task_manager import TaskManager, TaskStatus

MAX_NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1))

routes = Routes()


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    """
    Encode audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Encode the audio
        start_time = time.time()
        tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
        logger.info(
            f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms"
        )

        # Return the response
        return ormsgpack.packb(
            ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN encode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to encode audio"
        )


@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    """
    Decode tokens to audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Decode the audio
        tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
        start_time = time.time()
        audios = batch_vqgan_decode(decoder_model, tokens)
        logger.info(
            f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms"
        )
        audios = [audio.astype(np.float16).tobytes() for audio in audios]

        # Return the response
        return ormsgpack.packb(
            ServeVQGANDecodeResponse(audios=audios),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN decode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to decode tokens to audio"
        )


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    """
    Generate speech from text using TTS model.
    """
    try:
        # Get the model from the app
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine
        sample_rate = engine.decoder_model.sample_rate

        # Check if the text is too long
        if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Text is too long, max length is {app_state.max_text_length}",
            )

        # Check if streaming is enabled
        if req.streaming and req.format != "wav":
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Streaming only supports WAV format",
            )

        # Perform TTS
        if req.streaming:
            return StreamResponse(
                iterable=inference_async(req, engine),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )
        else:
            fake_audios = next(inference(req, engine))
            buffer = io.BytesIO()
            sf.write(
                buffer,
                fake_audios,
                sample_rate,
                format=req.format,
            )

            return StreamResponse(
                iterable=buffer_to_async_generator(buffer.getvalue()),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to generate speech"
        )


@routes.http.post("/v1/references/add")
async def add_reference(
    id: str = Body(...), audio: UploadFile = Body(...), text: str = Body(...)
):
    """
    Add a new reference voice with audio file and text.
    """
    temp_file_path = None

    try:
        # Validate input parameters
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")

        if not text or not text.strip():
            raise ValueError("Reference text cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Read the uploaded audio file
        audio_content = audio.read()
        if not audio_content:
            raise ValueError("Audio file is empty or could not be read")

        # Create a temporary file for the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        # Add the reference using the engine's reference loader
        engine.add_reference(id, temp_file_path, text)

        response = AddReferenceResponse(
            success=True,
            message=f"Reference voice '{id}' added successfully",
            reference_id=id,
        )
        return format_response(response)

    except FileExistsError as e:
        logger.warning(f"Reference ID '{id}' already exists: {e}")
        response = AddReferenceResponse(
            success=False,
            message=f"Reference ID '{id}' already exists",
            reference_id=id,
        )
        return format_response(response, status_code=409)  # Conflict

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{id}': {e}")
        response = AddReferenceResponse(success=False, message=str(e), reference_id=id)
        return format_response(response, status_code=400)

    except (FileNotFoundError, OSError) as e:
        logger.error(f"File system error for reference '{id}': {e}")
        response = AddReferenceResponse(
            success=False, message="File system error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error adding reference '{id}': {e}", exc_info=True)
        response = AddReferenceResponse(
            success=False, message="Internal server error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(
                    f"Failed to clean up temporary file {temp_file_path}: {e}"
                )


@routes.http.get("/v1/references/list")
async def list_references():
    """
    Get a list of all available reference voice IDs.
    """
    try:
        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Get the list of reference IDs
        reference_ids = engine.list_reference_ids()

        response = ListReferencesResponse(
            success=True,
            reference_ids=reference_ids,
            message=f"Found {len(reference_ids)} reference voices",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Unexpected error listing references: {e}", exc_info=True)
        response = ListReferencesResponse(
            success=False, reference_ids=[], message="Internal server error occurred"
        )
        return format_response(response, status_code=500)


@routes.http.delete("/v1/references/delete")
async def delete_reference(reference_id: str = Body(...)):
    """
    Delete a reference voice by ID.
    """
    try:
        # Validate input parameters
        if not reference_id or not reference_id.strip():
            raise ValueError("Reference ID cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Delete the reference using the engine's reference loader
        engine.delete_reference(reference_id)

        response = DeleteReferenceResponse(
            success=True,
            message=f"Reference voice '{reference_id}' deleted successfully",
            reference_id=reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(f"Reference ID '{reference_id}' not found: {e}")
        response = DeleteReferenceResponse(
            success=False,
            message=f"Reference ID '{reference_id}' not found",
            reference_id=reference_id,
        )
        return format_response(response, status_code=404)  # Not Found

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False, message=str(e), reference_id=reference_id
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error deleting reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False,
            message="File system error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(
            f"Unexpected error deleting reference '{reference_id}': {e}", exc_info=True
        )
        response = DeleteReferenceResponse(
            success=False,
            message="Internal server error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/references/update")
async def update_reference(
    old_reference_id: str = Body(...), new_reference_id: str = Body(...)
):
    """
    Rename a reference voice directory from old_reference_id to new_reference_id.
    """
    try:
        # Validate input parameters
        if not old_reference_id or not old_reference_id.strip():
            raise ValueError("Old reference ID cannot be empty")
        if not new_reference_id or not new_reference_id.strip():
            raise ValueError("New reference ID cannot be empty")
        if old_reference_id == new_reference_id:
            raise ValueError("New reference ID must be different from old reference ID")

        # Validate ID format per ReferenceLoader rules
        id_pattern = r"^[a-zA-Z0-9\-_ ]+$"
        if not re.match(id_pattern, new_reference_id) or len(new_reference_id) > 255:
            raise ValueError(
                "New reference ID contains invalid characters or is too long"
            )

        # Access engine to update caches after renaming
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        refs_base = Path("references")
        old_dir = refs_base / old_reference_id
        new_dir = refs_base / new_reference_id

        # Existence checks
        if not old_dir.exists() or not old_dir.is_dir():
            raise FileNotFoundError(f"Reference ID '{old_reference_id}' not found")
        if new_dir.exists():
            # Conflict: destination already exists
            response = UpdateReferenceResponse(
                success=False,
                message=f"Reference ID '{new_reference_id}' already exists",
                old_reference_id=old_reference_id,
                new_reference_id=new_reference_id,
            )
            return format_response(response, status_code=409)

        # Perform rename
        old_dir.rename(new_dir)

        # Update in-memory cache key if present
        if old_reference_id in engine.ref_by_id:
            engine.ref_by_id[new_reference_id] = engine.ref_by_id.pop(old_reference_id)

        response = UpdateReferenceResponse(
            success=True,
            message=(
                f"Reference voice renamed from '{old_reference_id}' to '{new_reference_id}' successfully"
            ),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(str(e))
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as e:
        logger.warning(f"Invalid input for update reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error renaming reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message="File system error occurred",
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error updating reference: {e}", exc_info=True)
        response = UpdateReferenceResponse(
            success=False,
            message="Internal server error occurred",
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=500)


@routes.http.post("/generate_audio_enhanced_async")
async def generate_audio_enhanced_async(
    req: Annotated[AsyncGenerateRequest, Body(exclusive=True)]
):
    """
    异步生成音频接口。
    接收stepId参数，立即返回，在后台开始生成。
    """
    try:
        # 获取模型管理器和任务管理器
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        task_manager: TaskManager = app_state.task_manager
        engine = model_manager.tts_inference_engine

        # 检查文本长度
        if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
            response = AsyncGenerateResponse(
                success=False,
                message=f"Text is too long, max length is {app_state.max_text_length}",
                step_id=req.step_id,
                status="failed",
            )
            return format_response(response, status_code=400)

        # 转换为ServeTTSRequest
        tts_request = ServeTTSRequest(
            text=req.text,
            chunk_length=req.chunk_length,
            format=req.format,
            references=req.references,
            reference_id=req.reference_id,
            seed=req.seed,
            use_memory_cache=req.use_memory_cache,
            normalize=req.normalize,
            streaming=False,  # 异步任务不支持流式
            max_new_tokens=req.max_new_tokens,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
        )

        # 创建任务
        try:
            task = task_manager.create_task(req.step_id, tts_request)
        except ValueError as e:
            response = AsyncGenerateResponse(
                success=False,
                message=str(e),
                step_id=req.step_id,
                status="failed",
            )
            return format_response(response, status_code=400)

        # 启动任务
        task_manager.start_task(task, engine)

        response = AsyncGenerateResponse(
            success=True,
            message="Task created and started",
            step_id=req.step_id,
            status=task.status.value,
        )
        return format_response(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in async generation: {e}", exc_info=True)
        response = AsyncGenerateResponse(
            success=False,
            message=f"Failed to create async task: {str(e)}",
            step_id=req.step_id if "req" in locals() else "unknown",
            status="failed",
        )
        return format_response(response, status_code=500)


@routes.http.get("/get_task_status")
async def get_task_status():
    """
    查询任务状态接口。
    返回对应stepId的生成状态，如果已完成，则提供下载地址。
    """
    try:
        # 从查询参数获取 step_id
        step_id = request.query_params.get("step_id")
        if not step_id:
            response = TaskStatusResponse(
                success=False,
                step_id="",
                status="error",
                created_at=0,
                error="step_id query parameter is required",
            )
            return format_response(response, status_code=400)
        
        app_state = request.app.state
        task_manager: TaskManager = app_state.task_manager

        task_info = task_manager.get_task_info(step_id)
        if task_info is None:
            response = TaskStatusResponse(
                success=False,
                step_id=step_id,
                status="not_found",
                created_at=0,
            )
            return format_response(response, status_code=404)

        response = TaskStatusResponse(
            success=True,
            step_id=task_info["step_id"],
            status=task_info["status"],
            created_at=task_info["created_at"],
            started_at=task_info.get("started_at"),
            completed_at=task_info.get("completed_at"),
            download_url=task_info.get("download_url"),
            error=task_info.get("error"),
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error getting task status: {e}", exc_info=True)
        # 尝试获取 step_id，如果获取失败则使用空字符串
        step_id = request.query_params.get("step_id", "")
        response = TaskStatusResponse(
            success=False,
            step_id=step_id,
            status="error",
            created_at=0,
            error=str(e),
        )
        return format_response(response, status_code=500)


@routes.http.get("/download_result")
async def download_result():
    """
    下载生成结果接口。
    返回对应stepId的生成结果文件。
    """
    try:
        # 从查询参数获取 step_id
        step_id = request.query_params.get("step_id")
        if not step_id:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="step_id query parameter is required",
            )
        
        app_state = request.app.state
        task_manager: TaskManager = app_state.task_manager

        task = task_manager.get_task(step_id)
        if task is None:
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                content=f"Task {step_id} not found",
            )

        if task.status != TaskStatus.COMPLETED:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Task {step_id} is not completed, current status: {task.status.value}",
            )

        if not task.result_path or not os.path.exists(task.result_path):
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                content=f"Result file for task {step_id} not found",
            )

        # 读取文件并返回
        with open(task.result_path, "rb") as f:
            file_content = f.read()

        # 获取文件格式
        file_format = task.request.format
        filename = f"audio_{step_id}.{file_format}"

        return StreamResponse(
            iterable=buffer_to_async_generator(file_content),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            },
            content_type=get_content_type(file_format),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading result: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content=f"Failed to download result: {str(e)}",
        )


@routes.http.post("/stop_generation")
async def stop_generation():
    """
    停止所有正在运行的生成任务。
    """
    try:
        app_state = request.app.state
        task_manager: TaskManager = app_state.task_manager

        count = task_manager.cancel_all_tasks()

        response = StopTaskResponse(
            success=True,
            message=f"Stopped {count} task(s)",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error stopping generation: {e}", exc_info=True)
        response = StopTaskResponse(
            success=False,
            message=f"Failed to stop generation: {str(e)}",
        )
        return format_response(response, status_code=500)


@routes.http.post("/stop_async_task/{step_id}")
async def stop_async_task(step_id: str):
    """
    停止指定的异步任务。
    """
    try:
        app_state = request.app.state
        task_manager: TaskManager = app_state.task_manager

        success = task_manager.cancel_task(step_id)
        if not success:
            response = StopTaskResponse(
                success=False,
                message=f"Task {step_id} not found",
                step_id=step_id,
            )
            return format_response(response, status_code=404)

        response = StopTaskResponse(
            success=True,
            message=f"Task {step_id} stopped",
            step_id=step_id,
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Error stopping task: {e}", exc_info=True)
        response = StopTaskResponse(
            success=False,
            message=f"Failed to stop task: {str(e)}",
            step_id=step_id,
        )
        return format_response(response, status_code=500)
