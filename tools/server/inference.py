from http import HTTPStatus

import numpy as np
from kui.asgi import HTTPException
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest

AMPLITUDE = 32768  # Needs an explaination


class PartialResultError(Exception):
    """部分结果错误：表示推理过程中发生错误，但已生成部分结果"""
    def __init__(self, error_message: str):
        self.error_message = error_message
        super().__init__(error_message)


def inference_wrapper(req: ServeTTSRequest, engine: TTSInferenceEngine):
    """
    Wrapper for the inference function.
    Used in the API server.
    """
    count = 0
    final_received = False  # 标记是否已收到 final 结果
    
    for result in engine.inference(req):
        match result.code:
            case "header":
                if isinstance(result.audio, tuple):
                    yield result.audio[1]

            case "error":
                # 如果已经收到 final 结果，抛出特殊的 PartialResultError 异常
                # 这样 task_manager 可以捕获并保存已生成的数据，状态保持为 processing
                if final_received:
                    # 已收到 final，抛出 PartialResultError，让 task_manager 知道发生了错误
                    # 但 task_manager 会检查 audio_data，如果有数据就继续保存
                    logger.warning(f"Error occurred after final result: {result.error}")
                    raise PartialResultError(str(result.error))
                else:
                    # 没有收到 final，抛出 HTTPException
                    raise HTTPException(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        content=str(result.error),
                    )

            case "segment":
                count += 1
                if isinstance(result.audio, tuple):
                    yield (result.audio[1] * AMPLITUDE).astype(np.int16).tobytes()

            case "final":
                count += 1
                final_received = True
                if isinstance(result.audio, tuple):
                    yield result.audio[1]
                # 不立即 return，继续处理可能的 error
                # 如果后续有 error，会抛出 PartialResultError（因为 final_received = True）

    if count == 0:
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )
