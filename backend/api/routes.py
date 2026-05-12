from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from services.image_inference import image_detector
from services.video_inference import video_detector
from services.audio_detector import audio_detector

router = APIRouter()

@router.post("/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    # Supported MIME types for conversion
    allowed_types = [
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/webm", "video/webm", # Browsers often send audio blobs as video/webm
        "audio/x-m4a", "audio/m4a", "audio/aac", "audio/mp4"
    ]
    
    if file.content_type not in allowed_types:
        # Check by extension if MIME is generic application/octet-stream
        ext = file.filename.split('.')[-1].lower() if file.filename else ""
        if ext not in ["wav", "mp3", "webm", "m4a", "aac"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Please upload WAV, MP3, WebM, or M4A."
            )
    
    try:
        contents = await file.read()
        # Heavy inference remains in threadpool
        result = await run_in_threadpool(audio_detector.analyze, contents, filename=file.filename)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return JSONResponse(content={
            "filename": file.filename,
            "type": "audio",
            "prediction": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...), heatmap: bool = False):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    contents = await file.read()
    # Run heavy inference in a threadpool to avoid blocking the event loop
    result = await run_in_threadpool(image_detector.analyze, contents, generate_heatmap=heatmap)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return JSONResponse(content={
        "filename": file.filename,
        "type": "image",
        "prediction": result
    })

@router.post("/detect/video")
async def detect_video(file: UploadFile = File(...), heatmap: bool = False):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    
    contents = await file.read()
    # Run heavy video analysis in a threadpool
    result = await run_in_threadpool(video_detector.analyze, contents, generate_heatmap=heatmap)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return JSONResponse(content={
        "filename": file.filename,
        "type": "video",
        "prediction": result
    })

@router.post("/detect/video/realtime")
async def detect_video_realtime(file: UploadFile = File(...)):
    """Ultra-fast detection for real-time webcam streams."""
    if not file.content_type.startswith("video/") and not file.content_type.startswith("application/octet-stream"):
         # Accept octet-stream for some blob uploads
         pass
    
    contents = await file.read()
    
    # Temporarily reduce frame count for real-time mode
    original_num_frames = video_detector.num_frames
    video_detector.num_frames = 8 # Ultra fast
    
    try:
        result = await run_in_threadpool(video_detector.analyze, contents)
    finally:
        video_detector.num_frames = original_num_frames
        
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return JSONResponse(content={
        "filename": file.filename,
        "type": "video_realtime",
        "prediction": result
    })
