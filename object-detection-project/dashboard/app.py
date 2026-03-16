"""
Streamlit Dashboard for the Object Detection System
Interactive UI for image, video, and webcam detection
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime

# Import project modules
from src.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DETECTED_IMAGES_DIR,
    DETECTED_VIDEOS_DIR,
    COCO_CLASSES,
)
from src.detect import DetectionEngine
from src.postprocess import PostProcessor, calculate_statistics, categorize_detections
from src.visualize import Visualizer
from src.tracker import create_tracker
from utils.metrics import DetectionMetrics


# Page configuration
st.set_page_config(
    page_title="AI Object Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_detection_engine(model_name: str = "yolov8s.pt"):
    """
    Load detection engine (cached).
    
    Args:
        model_name: YOLOv8 model variant
    
    Returns:
        DetectionEngine instance
    """
    engine = DetectionEngine(model_path=model_name)
    engine.load_model()
    return engine


def run_detection_on_image(image, engine, post_processor, visualizer):
    """
    Run detection on an uploaded image.
    
    Args:
        image: Input image (numpy array)
        engine: Detection engine
        post_processor: Post-processor
        visualizer: Visualizer
    
    Returns:
        Tuple of (annotated_image, stats, detections)
    """
    # Run detection
    detections = engine.detect(image)
    
    # Process results
    result = post_processor.process(detections, source_type='image', save_log=False)
    stats = result['stats']
    
    # Visualize
    annotated = visualizer.visualize(image, detections, stats)
    
    return annotated, stats, detections


def run_detection_on_video(video_path, engine, post_processor, visualizer, progress_bar):
    """
    Run detection on a video file.
    
    Args:
        video_path: Path to video file
        engine: Detection engine
        post_processor: Post-processor
        visualizer: Visualizer
        progress_bar: Streamlit progress bar
    
    Returns:
        Tuple of (output_video_path, final_stats)
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temp output file
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_output.name
    temp_output.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        detections = engine.detect(frame)
        all_detections.extend(detections)
        
        # Process results
        result = post_processor.process(detections, source_type='video', save_log=False)
        stats = result['stats']
        
        # Visualize
        annotated = visualizer.visualize(frame, detections, stats)
        
        # Write output
        out.write(annotated)
        
        frame_count += 1
        
        # Update progress
        if progress_bar:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
    
    cap.release()
    out.release()
    
    # Calculate final statistics
    final_stats = calculate_statistics(all_detections)
    final_stats['total_frames'] = frame_count
    final_stats['avg_detections_per_frame'] = len(all_detections) / max(frame_count, 1)
    
    return output_path, final_stats


def main():
    """Main dashboard function."""
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_options = {
        'YOLOv8 Nano (Fastest)': 'yolov8n.pt',
        'YOLOv8 Small (Balanced)': 'yolov8s.pt',
        'YOLOv8 Medium': 'yolov8m.pt',
        'YOLOv8 Large': 'yolov8l.pt',
        'YOLOv8 XLarge (Most Accurate)': 'yolov8x.pt',
    }
    
    selected_model = st.sidebar.selectbox(
        "Model",
        options=list(model_options.keys()),
        index=1,
    )
    model_path = model_options[selected_model]
    
    # Detection settings
    st.sidebar.subheader("Detection Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=0.1,
        max_value=0.9,
        value=IOU_THRESHOLD,
        step=0.05,
    )
    
    # Tracking option
    enable_tracking = st.sidebar.checkbox("Enable Object Tracking", value=False)
    
    # Main content
    st.title("🎯 AI Object Detection Dashboard")
    st.markdown("---")
    
    # Initialize components
    with st.spinner("Loading detection model..."):
        engine = load_detection_engine(model_path)
        engine.confidence_threshold = confidence_threshold
        engine.iou_threshold = iou_threshold
    
    post_processor = PostProcessor()
    visualizer = Visualizer()
    
    st.success("✅ Model loaded successfully!")
    
    # Input mode selection
    st.subheader("📥 Input Source")
    input_mode = st.radio(
        "Select input mode:",
        ["🖼️ Image Upload", "🎥 Video Upload", "📹 Webcam"],
        horizontal=True,
    )
    
    # Process based on input mode
    if input_mode == "🖼️ Image Upload":
        st.markdown("### Upload an image for detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with col1:
                st.image(image_rgb, caption="Original Image", use_container_width=True)
            
            # Run detection button
            if st.button("🔍 Run Detection", key="detect_image"):
                with st.spinner("Running detection..."):
                    annotated, stats, detections = run_detection_on_image(
                        image, engine, post_processor, visualizer
                    )
                    
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
                    
                    # Display statistics
                    st.markdown("### 📊 Detection Results")
                    
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        st.metric("Total Objects", stats['total_detections'])
                    
                    with metrics_col2:
                        st.metric("Persons", stats['person_count'])
                    
                    with metrics_col3:
                        st.metric("Vehicles", stats['vehicle_count'])
                    
                    with metrics_col4:
                        st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
                    
                    # Class breakdown
                    if stats['class_counts']:
                        st.markdown("#### Class Breakdown")
                        
                        class_data = {k: v for k, v in stats['class_counts'].items()}
                        st.bar_chart(class_data)
                    
                    # Detailed detections
                    with st.expander("📋 View Detailed Detections"):
                        for i, det in enumerate(detections):
                            st.write(f"**{i+1}. {det['class_name']}** - Confidence: {det['confidence']:.2%}")
    
    elif input_mode == "🎥 Video Upload":
        st.markdown("### Upload a video for detection")
        
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv'],
        )
        
        if uploaded_video is not None:
            # Save to temp file
            temp_video = tempfile.NamedTemporaryFile(suffix=Path(uploaded_video.name).suffix, delete=False)
            temp_video.write(uploaded_video.getvalue())
            temp_video_path = temp_video.name
            temp_video.close()
            
            # Display video info
            st.video(temp_video_path)
            
            # Run detection button
            if st.button("🔍 Run Detection on Video", key="detect_video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Processing video... This may take a while."):
                    status_text.text("Running object detection on each frame...")
                    
                    output_path, stats = run_detection_on_video(
                        temp_video_path,
                        engine,
                        post_processor,
                        visualizer,
                        progress_bar,
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    # Display results
                    st.markdown("### 📊 Detection Results")
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Total Detections", stats['total_detections'])
                    
                    with metrics_col2:
                        st.metric("Frames Processed", stats.get('total_frames', 0))
                    
                    with metrics_col3:
                        st.metric("Avg per Frame", f"{stats.get('avg_detections_per_frame', 0):.2f}")
                    
                    # Output video
                    st.markdown("### 🎬 Output Video")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Output Video",
                        data=video_bytes,
                        file_name=f"detected_{Path(uploaded_video.name).name}",
                        mime="video/mp4",
                    )
    
    elif input_mode == "📹 Webcam":
        st.markdown("### Real-time Webcam Detection")
        
        st.info("📌 **Note:** For real-time webcam detection, please use the command-line interface:")
        st.code("python main.py --source webcam", language="bash")
        
        st.markdown("""
        #### Keyboard Shortcuts:
        - `q` - Quit
        - `s` - Save snapshot
        
        The webcam mode provides:
        - ✅ Real-time detection at maximum FPS
        - ✅ Object tracking (if enabled)
        - ✅ Live statistics overlay
        - ✅ Snapshot saving capability
        """)
        
        # Placeholder for future webcam implementation
        # Note: Streamlit doesn't support real-time webcam processing well
        # due to its request-response architecture
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Built with YOLOv8 + OpenCV + Streamlit</p>
            <p>AI Object Detection System © 2024</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
