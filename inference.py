import argparse
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.image_inference import image_detector
from backend.services.video_inference import video_detector

def main():
    parser = argparse.ArgumentParser(description="DeepGuard Pro Inference CLI")
    parser.add_argument("--type", choices=['image', 'video'], required=True, help="Type of media to analyze")
    parser.add_argument("--file", required=True, help="Path to the media file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return
        
    with open(args.file, 'rb') as f:
        file_bytes = f.read()
        
    print(f"Analyzing {args.type}: {args.file}...")
    
    if args.type == 'image':
        result = image_detector.analyze(file_bytes)
    else:
        result = video_detector.analyze(file_bytes)
        
    if "error" in result:
        print(f"Analysis Failed: {result['error']}")
    else:
        print("\n--- Inference Results ---")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Details: {result['analysis']}")

if __name__ == "__main__":
    main()
