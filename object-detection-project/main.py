"""
Main entry point for the Object Detection System
Provides command-line interface for running detections
"""

import argparse
import sys
import cv2
from pathlib import Path

from src.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DETECTED_IMAGES_DIR,
    DETECTED_VIDEOS_DIR,
)
from src.preprocess import InputHandler, preprocess_frame
from src.detect import DetectionEngine
from src.postprocess import PostProcessor, calculate_statistics
from src.visualize import Visualizer
from src.tracker import create_tracker
from utils.metrics import DetectionMetrics
from utils.logger import log_info, log_error


def detect_image(
    image_path: str,
    engine: DetectionEngine,
    post_processor: PostProcessor,
    visualizer: Visualizer,
    output_dir: str = DETECTED_IMAGES_DIR,
):
    """
    Run detection on a single image.
    
    Args:
        image_path: Path to input image
        engine: Detection engine
        post_processor: Post-processor
        visualizer: Visualizer
        output_dir: Directory for output
    """
    log_info(f"Processing image: {image_path}")
    
    # Load image
    handler = InputHandler(image_path)
    if not handler.open():
        log_error(f"Failed to load image: {image_path}")
        return
    
    success, frame = handler.read()
    if not success:
        log_error(f"Failed to read image: {image_path}")
        return
    
    # Run detection
    detections = engine.detect(frame)
    
    # Process results
    result = post_processor.process(detections, source_type='image')
    stats = result['stats']
    
    # Visualize
    annotated = visualizer.visualize(frame, detections, stats)
    
    # Save output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"detected_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated)
    
    log_info(f"Output saved to: {output_path}")
    log_info(f"Detections: {stats['total_detections']} objects found")
    
    handler.release()


def detect_video(
    video_path: str,
    engine: DetectionEngine,
    post_processor: PostProcessor,
    visualizer: Visualizer,
    output_dir: str = DETECTED_VIDEOS_DIR,
    show_output: bool = False,
    use_tracking: bool = False,
):
    """
    Run detection on a video file.
    
    Args:
        video_path: Path to input video
        engine: Detection engine
        post_processor: Post-processor
        visualizer: Visualizer
        output_dir: Directory for output
        show_output: Whether to display output window
        use_tracking: Whether to use object tracking
    """
    log_info(f"Processing video: {video_path}")
    
    # Open video
    handler = InputHandler(video_path)
    if not handler.open():
        log_error(f"Failed to open video: {video_path}")
        return
    
    cap = handler.cap
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log_info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup video writer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"detected_{Path(video_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Initialize tracker if needed
    tracker = create_tracker(use_bytetrack=use_tracking) if use_tracking else None
    
    # Initialize metrics
    metrics = DetectionMetrics()
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Start inference timing
            metrics.start_inference()
            
            # Run detection
            detections = engine.detect(frame)
            
            # Stop inference timing
            inference_time = metrics.stop_inference()
            
            # Apply tracking if enabled
            if tracker and detections:
                detections = tracker.update(detections)
            
            # Record metrics
            metrics.record_frame(detections, inference_time)
            
            # Process results
            result = post_processor.process(detections, source_type='video', save_log=False)
            stats = result['stats']
            stats['fps'] = metrics.fps_counter.get_fps()
            
            # Visualize
            annotated = visualizer.visualize(
                frame, 
                detections, 
                stats,
                alerts=result['alerts'],
            )
            
            # Write output
            out.write(annotated)
            
            # Show output if requested
            if show_output:
                cv2.imshow('Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                log_info(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        log_info(f"Video processing complete!")
        
        # Print summary
        summary = metrics.get_summary()
        log_info(f"Average FPS: {summary['avg_fps']}")
        log_info(f"Average Inference Time: {summary['avg_inference_time_ms']:.2f}ms")
        log_info(f"Total Detections: {summary['total_detections']}")
        
    except KeyboardInterrupt:
        log_info("Interrupted by user")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        handler.release()
    
    log_info(f"Output saved to: {output_path}")


def detect_webcam(
    source: int = 0,
    engine: DetectionEngine = None,
    post_processor: PostProcessor = None,
    visualizer: Visualizer = None,
    show_output: bool = True,
    use_tracking: bool = False,
):
    """
    Run detection on webcam feed.
    
    Args:
        source: Webcam device index
        engine: Detection engine
        post_processor: Post-processor
        visualizer: Visualizer
        show_output: Whether to display output window
        use_tracking: Whether to use object tracking
    """
    log_info(f"Starting webcam detection (source: {source})")
    
    # Initialize components if not provided
    if engine is None:
        engine = DetectionEngine()
        engine.load_model()
    
    if post_processor is None:
        post_processor = PostProcessor()
    
    if visualizer is None:
        visualizer = Visualizer()
    
    # Open webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log_error(f"Failed to open webcam: {source}")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize tracker
    tracker = create_tracker(use_bytetrack=use_tracking) if use_tracking else None
    
    # Initialize metrics
    metrics = DetectionMetrics()
    
    log_info("Press 'q' to quit, 's' to save snapshot")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Start inference timing
            metrics.start_inference()
            
            # Run detection
            detections = engine.detect(frame)
            
            # Stop inference timing
            inference_time = metrics.stop_inference()
            
            # Apply tracking if enabled
            if tracker and detections:
                detections = tracker.update(detections)
            
            # Record metrics
            metrics.record_frame(detections, inference_time)
            
            # Process results
            result = post_processor.process(detections, source_type='webcam', save_log=False)
            stats = result['stats']
            stats['fps'] = metrics.fps_counter.get_fps()
            
            # Visualize
            annotated = visualizer.visualize(
                frame,
                detections,
                stats,
                alerts=result['alerts'],
            )
            
            # Show output
            if show_output:
                cv2.imshow('Webcam Detection', annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save snapshot
                    snapshot_path = Path(DETECTED_IMAGES_DIR) / f"snapshot_{metrics.total_frames}.jpg"
                    cv2.imwrite(str(snapshot_path), annotated)
                    log_info(f"Snapshot saved: {snapshot_path}")
    
    except KeyboardInterrupt:
        log_info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        summary = metrics.get_summary()
        log_info(f"\nSession Summary:")
        log_info(f"  Frames processed: {summary['total_frames']}")
        log_info(f"  Average FPS: {summary['avg_fps']}")
        log_info(f"  Total detections: {summary['total_detections']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Object Detection System using YOLOv8'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='webcam',
        help='Input source: image path, video path, or "webcam"'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_PATH,
        help='Path to YOLOv8 model weights'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help='Confidence threshold (0-1)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=IOU_THRESHOLD,
        help='IoU threshold for NMS (0-1)'
    )
    
    parser.add_argument(
        '--tracking',
        action='store_true',
        help='Enable object tracking'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable display window'
    )
    
    args = parser.parse_args()
    
    # Initialize detection engine
    log_info("Initializing detection system...")
    engine = DetectionEngine(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
    )
    
    if not engine.load_model():
        log_error("Failed to load model")
        sys.exit(1)
    
    # Initialize other components
    post_processor = PostProcessor()
    visualizer = Visualizer()
    
    # Route to appropriate handler
    if args.source.lower() == 'webcam':
        detect_webcam(
            engine=engine,
            post_processor=post_processor,
            visualizer=visualizer,
            show_output=not args.no_display,
            use_tracking=args.tracking,
        )
    elif Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        detect_image(
            image_path=args.source,
            engine=engine,
            post_processor=post_processor,
            visualizer=visualizer,
        )
    else:
        detect_video(
            video_path=args.source,
            engine=engine,
            post_processor=post_processor,
            visualizer=visualizer,
            show_output=not args.no_display,
            use_tracking=args.tracking,
        )


if __name__ == '__main__':
    main()
