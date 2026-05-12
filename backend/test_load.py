import torch
import sys
import os

# Mocking services to check initialization
try:
    print("Attempting to load VideoDeepfakeDetector...")
    from services.video_inference import video_detector
    print("VideoDeepfakeDetector loaded successfully.")
    
    print("Attempting to load ImageDeepfakeDetector...")
    from services.image_inference import image_detector
    print("ImageDeepfakeDetector loaded successfully.")
    
    print("\nEnvironment Check:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

except Exception as e:
    print(f"\nCRITICAL ERROR during initialization: {e}")
    import traceback
    traceback.print_exc()
