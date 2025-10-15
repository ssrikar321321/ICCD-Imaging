import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from skimage import exposure

# Page configuration
st.set_page_config(
    page_title="ICCD Image Processing Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #581c87 50%, #1e293b 100%);
    }
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        margin: 0.5rem 0 0 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #9333ea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Colormap definitions
COLORMAPS = {
    'jet': np.array([
        [0, 0, 143], [0, 0, 159], [0, 0, 175], [0, 0, 191], [0, 0, 207],
        [0, 0, 223], [0, 0, 239], [0, 0, 255], [0, 16, 255], [0, 32, 255],
        [0, 48, 255], [0, 64, 255], [0, 80, 255], [0, 96, 255], [0, 112, 255],
        [0, 128, 255], [0, 143, 255], [0, 159, 255], [0, 175, 255], [0, 191, 255],
        [0, 207, 255], [0, 223, 255], [0, 239, 255], [0, 255, 255], [16, 255, 239],
        [32, 255, 223], [48, 255, 207], [64, 255, 191], [80, 255, 175], [96, 255, 159],
        [112, 255, 143], [128, 255, 128], [143, 255, 112], [159, 255, 96], [175, 255, 80],
        [191, 255, 64], [207, 255, 48], [223, 255, 32], [239, 255, 16], [255, 255, 0],
        [255, 239, 0], [255, 223, 0], [255, 207, 0], [255, 191, 0], [255, 175, 0],
        [255, 159, 0], [255, 143, 0], [255, 128, 0], [255, 112, 0], [255, 96, 0],
        [255, 80, 0], [255, 64, 0], [255, 48, 0], [255, 32, 0], [255, 16, 0],
        [255, 0, 0], [239, 0, 0], [223, 0, 0], [207, 0, 0], [191, 0, 0],
        [175, 0, 0], [159, 0, 0], [143, 0, 0], [128, 0, 0]
    ], dtype=np.uint8),
    'inferno': np.array([
        [0, 0, 4], [20, 11, 53], [40, 17, 88], [60, 20, 115], [80, 20, 130],
        [100, 21, 140], [120, 24, 147], [140, 30, 150], [160, 38, 150], [180, 47, 148],
        [200, 57, 143], [220, 68, 135], [240, 80, 125], [252, 93, 113], [255, 107, 99],
        [255, 121, 84], [255, 136, 69], [255, 150, 54], [255, 165, 44], [255, 180, 38],
        [255, 195, 35], [255, 210, 37], [255, 225, 50], [255, 240, 75], [252, 255, 164]
    ] * 3, dtype=np.uint8),
    'hot': np.array([
        [0, 0, 0], [11, 0, 0], [21, 0, 0], [32, 0, 0], [43, 0, 0],
        [53, 0, 0], [64, 0, 0], [74, 0, 0], [85, 0, 0], [96, 0, 0],
        [106, 0, 0], [117, 0, 0], [128, 0, 0], [138, 0, 0], [149, 0, 0],
        [159, 0, 0], [170, 0, 0], [181, 0, 0], [191, 0, 0], [202, 0, 0],
        [213, 0, 0], [223, 0, 0], [234, 0, 0], [244, 0, 0], [255, 0, 0],
        [255, 11, 0], [255, 21, 0], [255, 32, 0], [255, 43, 0], [255, 53, 0],
        [255, 64, 0], [255, 74, 0], [255, 85, 0], [255, 96, 0], [255, 106, 0],
        [255, 117, 0], [255, 128, 0], [255, 138, 0], [255, 149, 0], [255, 159, 0],
        [255, 170, 0], [255, 181, 0], [255, 191, 0], [255, 202, 0], [255, 213, 0],
        [255, 223, 0], [255, 234, 0], [255, 244, 0], [255, 255, 0], [255, 255, 16],
        [255, 255, 32], [255, 255, 48], [255, 255, 64], [255, 255, 80], [255, 255, 96],
        [255, 255, 112], [255, 255, 128], [255, 255, 143], [255, 255, 159], [255, 255, 175],
        [255, 255, 191], [255, 255, 207], [255, 255, 223], [255, 255, 239], [255, 255, 255]
    ], dtype=np.uint8),
    'viridis': np.array([
        [68, 1, 84], [70, 8, 92], [71, 16, 99], [72, 23, 105], [72, 29, 111],
        [72, 35, 116], [72, 40, 120], [71, 46, 124], [70, 51, 127], [69, 56, 130],
        [68, 62, 133], [66, 67, 134], [65, 73, 137], [63, 78, 138], [61, 83, 139],
        [58, 88, 140], [56, 93, 141], [53, 99, 141], [51, 104, 142], [48, 110, 142],
        [46, 116, 142], [43, 121, 142], [41, 127, 142], [39, 132, 142], [37, 137, 142],
        [35, 143, 141], [33, 148, 140], [31, 154, 138], [31, 159, 136], [31, 165, 133],
        [33, 170, 131], [37, 175, 127], [42, 180, 124], [49, 185, 119], [56, 190, 113],
        [66, 195, 107], [78, 200, 100], [92, 205, 91], [108, 210, 81], [127, 215, 70],
        [147, 220, 50], [169, 224, 32], [192, 228, 21], [217, 232, 16], [243, 237, 16]
    ], dtype=np.uint8)
}

def apply_colormap(image_array, colormap_name):
    """Apply colormap to grayscale image"""
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2).astype(np.uint8)
    else:
        gray = image_array.astype(np.uint8)
    
    cmap = COLORMAPS[colormap_name]
    indices = (gray * (len(cmap) - 1) / 255).astype(int)
    colored = cmap[indices]
    
    return colored

def adjust_image(image_array, brightness, contrast):
    """Adjust brightness and contrast"""
    img = image_array.astype(float)
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    img = factor * (img - 128) + 128 + brightness
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def background_subtraction(measurement, background):
    """Subtract background from measurement image"""
    measurement_array = np.array(measurement.convert('L')).astype(np.int16)
    background_array = np.array(background.convert('L')).astype(np.int16)
    
    # Ensure same size
    if measurement_array.shape != background_array.shape:
        background = background.resize(measurement.size)
        background_array = np.array(background.convert('L')).astype(np.int16)
    
    subtracted = np.clip(measurement_array - background_array, 0, 255).astype(np.uint8)
    return subtracted

def enhance_contrast_clahe(image_array, clip_limit=0.005):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
    if image_array.max() > 1:
        image_array = image_array / 255.0
    
    enhanced = exposure.equalize_adapthist(image_array, clip_limit=clip_limit)
    return (enhanced * 255).astype(np.uint8)

def apply_threshold(image_array, threshold_value):
    """Apply thresholding to remove low intensity noise"""
    thresholded = image_array.copy()
    thresholded[thresholded < threshold_value] = 0
    return thresholded

def apply_gaussian_smoothing(image_array, sigma=2):
    """Apply Gaussian smoothing to reduce noise"""
    return gaussian_filter(image_array, sigma=sigma).astype(np.uint8)

def process_image(img, background_img, settings):
    """Process image with all settings"""
    img_array = np.array(img.convert('L'))
    
    # Apply crop if specified
    if settings.get('crop_coords'):
        x1, y1, x2, y2 = settings['crop_coords']
        img_array = img_array[y1:y2, x1:x2]
    
    # Background subtraction
    if background_img is not None and settings.get('use_bg_subtraction', False):
        bg_array = np.array(background_img.convert('L'))
        if settings.get('crop_coords'):
            bg_array = bg_array[y1:y2, x1:x2]
        
        if img_array.shape == bg_array.shape:
            img_array = np.clip(img_array.astype(np.int16) - bg_array.astype(np.int16), 0, 255).astype(np.uint8)
    
    # CLAHE enhancement
    if settings.get('use_clahe', False):
        img_array = enhance_contrast_clahe(img_array, settings.get('clahe_clip_limit', 0.005))
    
    # Gaussian smoothing
    if settings.get('use_smoothing', False):
        img_array = apply_gaussian_smoothing(img_array, settings.get('gaussian_sigma', 2))
    
    # Thresholding
    if settings.get('use_threshold', False):
        img_array = apply_threshold(img_array, settings.get('threshold_value', 5))
    
    # Adjust brightness and contrast
    img_array = adjust_image(img_array, settings['brightness'] - 100, settings['contrast'] - 100)
    
    # Apply colormap
    colored = apply_colormap(img_array, settings['colormap'])
    
    return Image.fromarray(colored)

# Initialize session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'background_image' not in st.session_state:
    st.session_state.background_image = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 ICCD Image Processing Dashboard</h1>
    <p>Advanced image processing with background subtraction, CLAHE, and thermal colormaps</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # File upload
    uploaded_files = st.file_uploader(
        "📤 Upload ICCD Images",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Background image upload
    background_file = st.file_uploader(
        "🖼️ Upload Background Image (Optional)",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        key="bg_uploader"
    )
    
    if background_file:
        st.session_state.background_image = Image.open(background_file)
        st.success("✅ Background image loaded!")
    
    st.divider()
    
    # Advanced Processing
    with st.expander("🔧 Advanced Processing", expanded=True):
        use_bg_subtraction = st.checkbox(
            "Background Subtraction",
            value=False,
            disabled=st.session_state.background_image is None,
            help="Subtract background image from measurements"
        )
        
        use_clahe = st.checkbox(
            "CLAHE Enhancement",
            value=False,
            help="Contrast Limited Adaptive Histogram Equalization"
        )
        
        if use_clahe:
            clahe_clip_limit = st.slider(
                "CLAHE Clip Limit",
                min_value=0.001,
                max_value=0.050,
                value=0.005,
                step=0.001,
                format="%.3f"
            )
        else:
            clahe_clip_limit = 0.005
        
        use_threshold = st.checkbox(
            "Thresholding",
            value=False,
            help="Remove low intensity noise"
        )
        
        if use_threshold:
            threshold_value = st.slider(
                "Threshold Value",
                min_value=0,
                max_value=50,
                value=5,
                help="Pixels below this value will be set to 0"
            )
        else:
            threshold_value = 5
        
        use_smoothing = st.checkbox(
            "Gaussian Smoothing",
            value=False,
            help="Apply Gaussian filter to reduce noise"
        )
        
        if use_smoothing:
            gaussian_sigma = st.slider(
                "Smoothing Sigma",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Higher values = more smoothing"
            )
        else:
            gaussian_sigma = 2.0
    
    st.divider()
    
    # Basic Processing
    colormap = st.selectbox(
        "🎨 Colormap",
        options=['jet', 'inferno', 'hot', 'viridis'],
        index=0,
        format_func=lambda x: x.capitalize() + (' (Thermal)' if x == 'jet' else ' (Plasma)' if x == 'inferno' else '')
    )
    
    brightness = st.slider("💡 Brightness", 0, 200, 100, help="100 = original")
    contrast = st.slider("🔆 Contrast", 0, 200, 100, help="100 = original")
    
    st.divider()
    
    # Layout controls
    grid_cols = st.selectbox("📐 Grid Columns", [2, 3, 4, 5], index=2)
    show_labels = st.checkbox("🏷️ Show Time Labels", value=True)
    
    st.divider()
    
    # Actions
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.uploaded_images = []
        st.session_state.processed_images = []
        st.session_state.background_image = None
        st.rerun()

# Process uploaded files
if uploaded_files:
    if len(uploaded_files) != len(st.session_state.uploaded_images):
        st.session_state.uploaded_images = []
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            st.session_state.uploaded_images.append({
                'name': file.name,
                'image': img,
                'timestamp': f"Time = {i * 20 + 58} ns"
            })

# Process images with current settings
if st.session_state.uploaded_images:
    settings = {
        'colormap': colormap,
        'brightness': brightness,
        'contrast': contrast,
        'use_bg_subtraction': use_bg_subtraction,
        'use_clahe': use_clahe,
        'clahe_clip_limit': clahe_clip_limit,
        'use_threshold': use_threshold,
        'threshold_value': threshold_value,
        'use_smoothing': use_smoothing,
        'gaussian_sigma': gaussian_sigma
    }
    
    st.session_state.processed_images = []
    for img_data in st.session_state.uploaded_images:
        processed = process_image(
            img_data['image'],
            st.session_state.background_image,
            settings
        )
        st.session_state.processed_images.append({
            'name': img_data['name'],
            'image': processed,
            'timestamp': img_data['timestamp']
        })

# Display metrics
if st.session_state.processed_images:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.processed_images)}</div>
            <div class="metric-label">Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        bg_status = "✅" if st.session_state.background_image else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{bg_status}</div>
            <div class="metric-label">Background</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        processing_count = sum([use_bg_subtraction, use_clahe, use_threshold, use_smoothing])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{processing_count}</div>
            <div class="metric-label">Filters Active</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{colormap.capitalize()}</div>
            <div class="metric-label">Colormap</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{grid_cols}</div>
            <div class="metric-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

# Display images in grid
if st.session_state.processed_images:
    rows = (len(st.session_state.processed_images) + grid_cols - 1) // grid_cols
    
    for row in range(rows):
        cols = st.columns(grid_cols)
        for col_idx in range(grid_cols):
            img_idx = row * grid_cols + col_idx
            if img_idx < len(st.session_state.processed_images):
                with cols[col_idx]:
                    img_data = st.session_state.processed_images[img_idx]
                    
                    caption = img_data['timestamp'] if show_labels else ""
                    st.image(img_data['image'], caption=caption, use_container_width=True)
                    
                    # Download button
                    buffered = io.BytesIO()
                    img_data['image'].save(buffered, format="PNG")
                    st.download_button(
                        label="⬇️ Download",
                        data=buffered.getvalue(),
                        file_name=f"processed_{img_data['name']}",
                        mime="image/png",
                        key=f"download_{img_idx}",
                        use_container_width=True
                    )
else:
    st.markdown("""
    <div style='text-align: center; padding: 4rem; background: rgba(255, 255, 255, 0.1); 
                backdrop-filter: blur(10px); border-radius: 1rem; border: 2px dashed rgba(255, 255, 255, 0.3);'>
        <h2 style='color: white; margin-bottom: 1rem;'>📤 No Images Loaded</h2>
        <p style='color: rgba(255, 255, 255, 0.7); font-size: 1.1rem;'>
            Upload your ICCD images using the sidebar to begin processing
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: rgba(255, 255, 255, 0.5); padding: 1rem;'>
    <small>ICCD Advanced Image Processing Dashboard | Powered by Streamlit</small>
</div>
""", unsafe_allow_html=True)
