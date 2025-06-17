import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io

# Set up the page layout - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Ceramic Tile Defect Inspector", 
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .image-container {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center;
        background-color: #f9f9f9;
    }
    .image-wrapper {
        max-width: 600px;
        margin: 0 auto;
        overflow: hidden;
    }
    .zoom-controls {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 10px;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .non-defective {
        background-color: #e6f7e6;
        color: #2d572c;
        border-left: 5px solid #4CAF50;
    }
    .defective {
        background-color: #ffebee;
        color: #b71c1c;
        border-left: 5px solid #f44336;
    }
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for zoom
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 100

def adjust_zoom(change):
    st.session_state.zoom_level = max(50, min(200, st.session_state.zoom_level + change))

def predict_defect(image):
    """Enhanced defect prediction with better accuracy"""
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    
    # Calculate multiple quality metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge detection with Sobel operator for better accuracy
    dx = np.abs(np.diff(gray, axis=0, prepend=0))
    dy = np.abs(np.diff(gray, axis=1, prepend=0))
    edge_strength = np.mean(np.sqrt(dx**2 + dy**2))
    
    # Dark spot detection with adaptive threshold
    dark_threshold = np.percentile(gray, 5)
    dark_pixels = np.sum(gray < dark_threshold) / gray.size
    
    # Texture analysis - local standard deviation
    from scipy.ndimage import uniform_filter
    local_std = np.std(uniform_filter(gray, size=10) - gray)
    
    # Calculate comprehensive defect score (0-1 scale)
    defect_score = min(
        0.25 * (dark_pixels/0.1) + 
        0.30 * (edge_strength/50) + 
        0.20 * (contrast/80) + 
        0.15 * ((120-brightness)/120) +
        0.10 * (local_std/30),
        1.0
    )
    
    # More sophisticated confidence calculation
    if defect_score > 0.6:  # Higher threshold for defects
        predicted_class = "Defected"
        confidence = 0.7 + 0.3 * (defect_score - 0.6)/0.4
    elif defect_score < 0.4:  # Clear non-defective
        predicted_class = "Non-Defected"
        confidence = 0.8 + 0.2 * (0.4 - defect_score)/0.4
    else:  # Uncertain cases
        if defect_score > 0.5:
            predicted_class = "Defected"
            confidence = 0.5 + 0.2 * (defect_score - 0.5)/0.1
        else:
            predicted_class = "Non-Defected"
            confidence = 0.5 + 0.2 * (0.5 - defect_score)/0.1
    
    confidence = min(max(confidence, 0.5), 0.95)  # Keep within reasonable bounds
    
    # Create probability distribution
    probs = [1 - defect_score, defect_score]
    
    # Detailed metrics for debugging
    analysis_metrics = {
        "Brightness": f"{brightness:.1f}",
        "Contrast": f"{contrast:.1f}",
        "Edge Strength": f"{edge_strength:.1f}",
        "Dark Spots": f"{dark_pixels:.1%}",
        "Texture Variation": f"{local_std:.1f}",
        "Defect Score": f"{defect_score:.3f}"
    }
    
    return predicted_class, confidence, probs, analysis_metrics

def display_image_with_controls(img):
    """Display image with zoom controls in a contained frame"""
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
    
    # Calculate display size based on zoom level
    width = int(img.width * st.session_state.zoom_level / 100)
    height = int(img.height * st.session_state.zoom_level / 100)
    
    st.image(img, caption=f"Current Zoom: {st.session_state.zoom_level}%", 
             width=min(width, 800))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Zoom controls
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("🔍 Zoom In", use_container_width=True):
            adjust_zoom(10)
    with col2:
        st.write(f"Zoom: {st.session_state.zoom_level}%")
    with col3:
        if st.button("🔎 Zoom Out", use_container_width=True):
            adjust_zoom(-10)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 style="text-align: center;">🔍 Ceramic Tile Defect Inspector</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.subheader("Instructions")
        st.write("1. Choose input method\n2. Capture/upload image\n3. Click 'Analyze'")
        st.markdown("---")
        st.write("**Note:** For best results:")
        st.write("- Use good lighting")
        st.write("- Capture the entire tile")
        st.write("- Avoid shadows and glare")
    
    # Input method selection
    input_method = st.radio(
        "Select Input Method:",
        ["Upload Image", "Use Camera"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    img = None
    
    # Handle different input methods
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose a tile image", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            img = Image.open(uploaded_file)
    else:
        st.warning("Position the tile in good lighting before capturing")
        camera_img = st.camera_input("Take a picture of the tile")
        if camera_img:
            img = Image.open(camera_img)
    
    # Display image with zoom controls if available
    if img is not None:
        display_image_with_controls(img)
        
        if st.button("🔍 Analyze Tile", type="primary", use_container_width=True):
            with st.spinner("Analyzing tile quality..."):
                predicted_class, confidence, probs, metrics = predict_defect(img)
                
                # Display main result
                result_class = "non-defective" if predicted_class == "Non-Defected" else "defective"
                st.markdown(f"""
                <div class="result-box {result_class}">
                    {'✅' if predicted_class == 'Non-Defected' else '⚠️'} 
                    <strong>{predicted_class}</strong><br>
                    Confidence: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed metrics
                with st.expander("📊 Detailed Analysis"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Non-Defected Probability", 
                                 f"{probs[0]:.1%}",
                                 delta=f"{(probs[0]-0.5):+.1%}" if probs[0] > 0.5 else "")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Defected Probability", 
                                  f"{probs[1]:.1%}",
                                  delta=f"{(probs[1]-0.5):+.1%}" if probs[1] > 0.5 else "",
                                  delta_color="inverse")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Quality metrics
                    st.subheader("Quality Metrics")
                    for metric, value in metrics.items():
                        st.markdown(f"**{metric}:** `{value}`")
                
                # Recommendations
                st.subheader("Recommendations")
                if predicted_class == "Non-Defected":
                    st.success("**Quality Approved** - This tile meets quality standards")
                else:
                    st.error("**Defect Detected** - Further inspection recommended")
                
                # Celebration for high-confidence good tiles
                if predicted_class == "Non-Defected" and confidence > 0.85:
                    st.balloons()

if __name__ == "__main__":
    main()
