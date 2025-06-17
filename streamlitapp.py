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

st.title("🔍 Ceramic Tile Defect Inspector")
st.caption("AI-Powered Tile Quality Detection - Identify Defective vs Non-Defective Tiles")

st.write(
    "Upload an image of a ceramic tile and our AI model will classify it as "
    "Non-Defected or Defected with confidence scores."
)

# Sidebar information
with st.sidebar:
    st.subheader("About Tile Defect Inspector")
    st.write(
        "This application helps identify manufacturing defects in ceramic tiles "
        "using computer vision. The model can detect various types of defects "
        "including cracks, chips, color inconsistencies, and surface imperfections."
    )
    
    st.write(
        "Simply upload an image of a tile and our AI will analyze it for defects."
    )
    
    st.markdown("---")
    st.markdown("**Supported file types:** JPG, JPEG, PNG")
    st.markdown("**Model classes:** Non-Defected, Defected")

# Simple rule-based prediction function for demonstration
def predict_defect(image):
    """
    Simple rule-based prediction - analyzes image characteristics
    Replace this with your actual model integration
    """
    # Convert image to numpy array for analysis
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Calculate image properties
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge detection (simple vertical/horizontal differences)
    edge_variation = (np.sum(np.abs(np.diff(gray, axis=0))) + 
                     np.sum(np.abs(np.diff(gray, axis=1)))) / gray.size
    
    # Dark spot detection (potential defects)
    dark_pixels = np.sum(gray < 50) / gray.size
    
    # Simple scoring system
    defect_score = 0
    
    if dark_pixels > 0.05:  # More than 5% dark pixels
        defect_score += 0.4
    
    if edge_variation > 25:  # High edge variation
        defect_score += 0.3
    
    if contrast > 70:  # High contrast
        defect_score += 0.2
    
    if brightness < 100:  # Low brightness
        defect_score += 0.1
    
    # Determine prediction
    if defect_score > 0.5:
        predicted_class = "Defected"
        confidence = min(0.65 + defect_score * 0.3, 0.95)
    else:
        predicted_class = "Non-Defected"
        confidence = max(0.7 + (1 - defect_score) * 0.25, 0.7)
    
    # Create probability distribution
    if predicted_class == "Non-Defected":
        non_defect_prob = confidence
        defect_prob = 1 - confidence
    else:
        defect_prob = confidence
        non_defect_prob = 1 - confidence
    
    return predicted_class, confidence, [non_defect_prob, defect_prob]

# Main app functionality
def main():
    st.subheader("Upload Tile Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Tile Image", width=400)
        st.success("Image uploaded successfully!")
        
        # Prediction button
        if st.button("Analyze Tile for Defects", type="primary"):
            with st.spinner("Analyzing tile for defects..."):
                predicted_class, confidence, probs = predict_defect(img)
                
                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                if predicted_class == "Non-Defected":
                    col1.success(f"✅ **{predicted_class}**")
                    col1.success(f"Confidence: {confidence:.2%}")
                    st.balloons()
                else:
                    col1.error(f"⚠️ **{predicted_class}**")
                    col1.error(f"Confidence: {confidence:.2%}")
                
                # Show detailed predictions
                st.subheader("Detailed Analysis")
                cols = st.columns(2)
                cols[0].metric("Non-Defected", f"{probs[0]:.2%}")
                cols[1].metric("Defected", f"{probs[1]:.2%}")
                
                # Recommendations
                st.subheader("Recommendations")
                if predicted_class == "Non-Defected":
                    st.info("This tile appears to be in good condition. No defects detected.")
                else:
                    st.warning("Defects detected! This tile should be inspected further or rejected.")

if __name__ == "__main__":
    main()
