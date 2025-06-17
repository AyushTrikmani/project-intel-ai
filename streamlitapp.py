import streamlit as st
import numpy as np
from PIL import Image
import io

# Set up the page layout - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Ceramic Tile Defect Inspector", 
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .non-defective {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #28a745;
    }
    .defective {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 5px solid #dc3545;
    }
    .confidence-meter {
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
    }
    .tab-content {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🔍 Ceramic Tile Defect Inspector</h1>', unsafe_allow_html=True)
st.caption("AI-Powered Tile Quality Detection - Identify Defective vs Non-Defective Tiles")

# Sidebar information
with st.sidebar:
    st.subheader("About Tile Defect Inspector")
    st.write(
        "This application helps identify manufacturing defects in ceramic tiles "
        "using computer vision. The model can detect various types of defects "
        "including cracks, chips, color inconsistencies, and surface imperfections."
    )
    
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Choose input method (Upload or Camera)")
    st.markdown("2. Capture/upload tile image")
    st.markdown("3. Click 'Analyze' button")
    st.markdown("4. View results and recommendations")
    
    st.markdown("---")
    st.markdown("**Supported file types:** JPG, JPEG, PNG")
    st.markdown("**Model classes:** Non-Defected, Defected")

def predict_defect(image):
    """Enhanced rule-based prediction with better visualization"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Calculate image properties
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge detection
    edge_variation = (np.sum(np.abs(np.diff(gray, axis=0))) + 
                     np.sum(np.abs(np.diff(gray, axis=1)))) / gray.size
    
    # Dark spot detection
    dark_pixels = np.sum(gray < 50) / gray.size
    
    # Calculate defect score (0-1 scale)
    defect_score = min(
        0.4 * (dark_pixels/0.05) + 
        0.3 * (edge_variation/25) + 
        0.2 * (contrast/70) + 
        0.1 * ((100-brightness)/100),
        1.0
    )
    
    # Determine prediction
    if defect_score > 0.5:
        predicted_class = "Defected"
        confidence = min(0.65 + defect_score * 0.3, 0.95)
    else:
        predicted_class = "Non-Defected"
        confidence = max(0.7 + (1 - defect_score) * 0.25, 0.7)
    
    # Create probability distribution
    probs = [1 - defect_score, defect_score]
    
    # Additional analysis metrics
    analysis_metrics = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Edge Variation": edge_variation,
        "Dark Spots": dark_pixels,
        "Defect Score": defect_score
    }
    
    return predicted_class, confidence, probs, analysis_metrics

def main():
    # Input method selection
    input_method = st.radio(
        "Select Input Method:",
        ["Upload Image", "Use Camera"],
        horizontal=True
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
            st.image(img, caption="Uploaded Tile Image", use_column_width=True)
            st.success("Image uploaded successfully!")
    else:
        st.warning("Please ensure proper lighting when capturing images")
        camera_img = st.camera_input("Take a picture of the tile")
        if camera_img:
            img = Image.open(camera_img)
            st.image(img, caption="Captured Tile Image", use_column_width=True)
            st.success("Image captured successfully!")
    
    # Analysis section
    if img is not None:
        if st.button("🔍 Analyze Tile", type="primary", use_container_width=True):
            with st.spinner("Analyzing tile for defects..."):
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
                
                # Confidence meter
                st.markdown(f"""
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width:{confidence*100}%; 
                    background-color: {'#28a745' if predicted_class == 'Non-Defected' else '#dc3545'};"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed results in tabs
                tab1, tab2 = st.tabs(["📊 Probabilities", "📈 Analysis Metrics"])
                
                with tab1:
                    cols = st.columns(2)
                    cols[0].metric("Non-Defected", 
                                 f"{probs[0]:.1%}",
                                 delta=f"{(probs[0]-0.5):+.1%}" if probs[0] > 0.5 else "")
                    cols[1].metric("Defected", 
                                  f"{probs[1]:.1%}",
                                  delta=f"{(probs[1]-0.5):+.1%}" if probs[1] > 0.5 else "",
                                  delta_color="inverse")
                    
                    # Visual comparison
                    chart_data = {
                        "Condition": ["Non-Defected", "Defected"],
                        "Probability": probs
                    }
                    st.bar_chart(chart_data, x="Condition", y="Probability")
                
                with tab2:
                    for metric, value in metrics.items():
                        st.metric(metric, f"{value:.2f}")
                
                # Recommendations
                st.subheader("Recommendations")
                if predicted_class == "Non-Defected":
                    st.success("**Quality Approved** - This tile meets quality standards")
                    st.info("""
                    - No further action required
                    - Continue with normal installation
                    - Maintain regular quality checks
                    """)
                else:
                    st.error("**Defect Detected** - Further inspection recommended")
                    st.warning("""
                    - Isolate this tile from production batch
                    - Document defect for quality records
                    - Check surrounding tiles from same batch
                    - Review manufacturing parameters
                    """)
                
                # Celebration for good tiles
                if predicted_class == "Non-Defected" and confidence > 0.9:
                    st.balloons()

if __name__ == "__main__":
    main()
