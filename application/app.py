#!/usr/bin/env python3
"""
TOKKATOT AI Cloud Application - Entry Point
Single command to run live webcam, demo, or video processing.

Quick start:
  python app.py              # Live webcam
  python app.py --demo       # Single image demo
  python app.py --video file.mp4  # Process video
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_parser():
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="TOKKATOT Edge - Real-Time Disease Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python app.py                                     # Real-time webcam
  python app.py --demo --image photo.jpg           # Demo mode (analyze image)
  python app.py --video video.mp4                  # Process video file
  python app.py --anomaly-threshold 5              # Sensitive (alert at 5%)
  python app.py --conf-threshold 0.7               # Strict detection
        """
    )
    
    parser.add_argument("--demo", action='store_true', help="Demo mode (analyze uploaded image)") 
    parser.add_argument("--image", type=str, help="Image file for demo mode analysis")
    parser.add_argument("--video", type=str, help="Video file to process")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam ID")
    parser.add_argument("--ensemble-model", type=str, default="application/inferences/ensemble_model.pth")
    parser.add_argument("--yolo-model", type=str, default="application/inferences/yolov8_custom.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="YOLO confidence")
    parser.add_argument("--anomaly-threshold", type=float, default=10.0, help="Alert threshold (percent)")
    parser.add_argument("--cloud-api", type=str, default="http://localhost:8000/api/v1", help="Cloud API URL")
    parser.add_argument("--no-cloud", action='store_true', help="Disable cloud verification")
    parser.add_argument("--device", default='auto', choices=['auto', 'cuda', 'cpu'])
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Lazy import EdgeApp (avoids hanging on startup)
    from backend.core import EdgeApp
    
    # Validate models
    if not Path(args.ensemble_model).exists():
        print(f"✗ Ensemble model not found: {args.ensemble_model}")
        sys.exit(1)
    if not Path(args.yolo_model).exists():
        print(f"✗ YOLO model not found: {args.yolo_model}")
        sys.exit(1)
    
    # Initialize and run
    try:
        app = EdgeApp(
            ensemble_model_path=args.ensemble_model,
            yolo_model_path=args.yolo_model,
            conf_threshold=args.conf_threshold,
            anomaly_threshold_pct=args.anomaly_threshold,
            cloud_api_url=None if args.no_cloud else args.cloud_api,
            device=args.device
        )
        
        if args.demo:
            if not args.image:
                print("✗ Demo mode requires --image argument")
                print("Example: python app.py --demo --image photo.jpg")
                sys.exit(1)
            app.capture_and_analyze(args.image)
        elif args.video:
            app.run_video_file(args.video)
        else:
            app.run_webcam(args.camera_id)
    
    except KeyboardInterrupt:
        print("\n✓ Stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
