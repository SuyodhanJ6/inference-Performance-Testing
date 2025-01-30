# from fastapi import FastAPI, BackgroundTasks
# from vllm import LLM, SamplingParams
# import uvicorn
# from pydantic import BaseModel
# import psutil
# import GPUtil
# import time
# from typing import Optional, List

# class GenerateRequest(BaseModel):
#     prompt: str
#     max_tokens: int = 512
#     temperature: float = 0.7
#     top_p: float = 0.95
#     stream: bool = False

# class ServerStats(BaseModel):
#     gpu_utilization: float
#     gpu_memory_used: float
#     gpu_memory_total: float
#     cpu_percent: float
#     ram_percent: float
#     active_requests: int
#     total_requests_served: int

# app = FastAPI()

# # Global statistics
# stats = {
#     "active_requests": 0,
#     "total_requests_served": 0
# }

# # Initialize LLM with optimized settings for RTX 4090
# model = LLM(
#     model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.90,
#     max_num_batched_tokens=8192,  # Adjust based on your VRAM
#     trust_remote_code=True  # Required for some models like Qwen
# )

# @app.get("/stats")
# async def get_stats():
#     gpus = GPUtil.getGPUs()
#     gpu = gpus[0]  # Assuming using first GPU
    
#     return ServerStats(
#         gpu_utilization=gpu.load * 100,
#         gpu_memory_used=gpu.memoryUsed,
#         gpu_memory_total=gpu.memoryTotal,
#         cpu_percent=psutil.cpu_percent(),
#         ram_percent=psutil.virtual_memory().percent,
#         active_requests=stats["active_requests"],
#         total_requests_served=stats["total_requests_served"]
#     )

# @app.post("/generate")
# async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
#     stats["active_requests"] += 1
#     try:
#         sampling_params = SamplingParams(
#             temperature=request.temperature,
#             top_p=request.top_p,
#             max_tokens=request.max_tokens
#         )
        
#         # Run in the default thread pool to avoid blocking
#         outputs = model.generate([request.prompt], sampling_params)
#         stats["total_requests_served"] += 1
#         return {"generated_text": outputs[0].outputs[0].text}
#     finally:
#         stats["active_requests"] -= 1

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, BackgroundTasks, HTTPException
from vllm import LLM, SamplingParams
import uvicorn
from pydantic import BaseModel
import psutil
import GPUtil
import time
from typing import Optional, List
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import asyncio
from sse_starlette.sse import EventSourceResponse

# --- Configuration Constants ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
GPU_MEMORY_UTILIZATION = 0.85
MAX_BATCHED_TOKENS = 16384
MAX_MODEL_LEN = 8192

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False

class ServerStats(BaseModel):
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_percent: float
    ram_percent: float
    active_requests: int
    total_requests_served: int
    avg_generation_time: float

# --- Global State ---
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=8)

stats = {
    "active_requests": 0,
    "total_requests_served": 0,
    "total_processing_time": 0.0,
    "lock": Lock()
}

# --- Model Initialization ---
model = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_num_batched_tokens=MAX_BATCHED_TOKENS,
    max_model_len=MAX_MODEL_LEN,
    enforce_eager=True,
    trust_remote_code=True
)

# --- Helper Functions ---
async def async_generate(prompt: str, params: SamplingParams):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model.generate([prompt], params)
    )

def get_gpu_stats():
    gpus = GPUtil.getGPUs()
    return gpus[0] if gpus else None

# --- API Endpoints ---
@app.get("/stats", response_model=ServerStats)
async def get_stats():
    gpu = get_gpu_stats()
    with stats["lock"]:
        avg_time = stats["total_processing_time"] / stats["total_requests_served"] \
            if stats["total_requests_served"] > 0 else 0
        
        return ServerStats(
            gpu_utilization=gpu.load * 100 if gpu else 0,
            gpu_memory_used=gpu.memoryUsed if gpu else 0,
            gpu_memory_total=gpu.memoryTotal if gpu else 0,
            cpu_percent=psutil.cpu_percent(),
            ram_percent=psutil.virtual_memory().percent,
            active_requests=stats["active_requests"],
            total_requests_served=stats["total_requests_served"],
            avg_generation_time=avg_time
        )

@app.post("/generate")
async def generate(request: GenerateRequest):
    start_time = time.time()
    
    try:
        # Update request tracking
        with stats["lock"]:
            stats["active_requests"] += 1

        # Validate input
        if len(request.prompt) > MAX_MODEL_LEN:
            raise HTTPException(400, "Prompt exceeds maximum length")

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )

        # Handle streaming
        if request.stream:
            async def stream_generator():
                try:
                    for output in model.generate_stream([request.prompt], sampling_params):
                        yield {"text": output.outputs[0].text}
                finally:
                    with stats["lock"]:
                        stats["active_requests"] -= 1
                        stats["total_requests_served"] += 1
                        stats["total_processing_time"] += time.time() - start_time
                        
            return EventSourceResponse(stream_generator())

        # Regular generation
        outputs = await async_generate(request.prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text

        return {"generated_text": generated_text}
    
    except Exception as e:
        raise HTTPException(500, str(e))
    
    finally:
        with stats["lock"]:
            stats["active_requests"] -= 1
            if not request.stream:
                stats["total_requests_served"] += 1
                stats["total_processing_time"] += time.time() - start_time

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,
        limit_concurrency=100
    )