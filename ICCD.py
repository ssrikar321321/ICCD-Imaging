import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from scipy.ndimage import gaussian_filter
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Page configuration
st.set_page_config(
    page_title="ICCD Image Processing Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

def add_timestamp_overlay(image: Image.Image, text: str) -> Image.Image:
    """Return a copy of the image with the timestamp text rendered on top."""

    if not text:
        return image

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a truetype font for better scaling, fall back to default otherwise
    font_size = max(12, base.width // 20)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    padding = max(8, font_size // 4)
    x = padding
    y = base.height - text_height - padding
    rectangle_coords = [
        x - padding,
        y - padding,
        x + text_width + padding,
        y + text_height + padding,
    ]

    draw.rectangle(rectangle_coords, fill=(0, 0, 0, 160))
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")


def process_iccd_image(img, background_img, settings):
    """Process ICCD image - SIMPLIFIED AND FIXED"""
    
    # Convert to grayscale numpy array
    if img.mode != 'L':
        img = img.convert('L')
    img_array = np.array(img, dtype=np.float64)
    
    # Apply crop if specified
    if settings.get('crop_coords'):
        y1, y2, x1, x2 = settings['crop_coords']
        img_array = img_array[y1:y2, x1:x2]
    
    # Background subtraction
    if background_img is not None and settings.get('use_bg_subtraction', False):
        bg_array = np.array(background_img.convert('L'), dtype=np.float64)
        if settings.get('crop_coords'):
            bg_array = bg_array[y1:y2, x1:x2]
        
        if img_array.shape == bg_array.shape:
            img_array = np.maximum(img_array - bg_array, 0)
    
    # CLAHE enhancement
    if settings.get('use_clahe', False):
        img_normalized = img_array / 255.0
        img_normalized = exposure.equalize_adapthist(
            img_normalized, 
            clip_limit=settings.get('clahe_clip_limit', 0.03)
        )
        img_array = img_normalized * 255.0
    
    # Gaussian smoothing
    if settings.get('use_smoothing', False):
        img_array = gaussian_filter(img_array, sigma=settings.get('gaussian_sigma', 2))
    
    # Thresholding
    if settings.get('use_threshold', False):
        threshold = settings.get('threshold_value', 5)
        img_array[img_array < threshold] = 0
    
    # Normalize to 0-1 range using percentiles - THIS IS THE KEY!
    p_low = np.percentile(img_array, settings.get('percentile_low', 1))
    p_high = np.percentile(img_array, settings.get('percentile_high', 99))
    
    if p_high > p_low:
        img_array = (img_array - p_low) / (p_high - p_low)
        img_array = np.clip(img_array, 0, 1)
    else:
        img_array = img_array / (img_array.max() + 1e-10)
    
    # Apply brightness and contrast
    brightness = (settings['brightness'] - 100) / 100.0
    contrast = (settings['contrast'] - 100) / 100.0
    
    img_array = img_array + brightness
    if contrast != 0:
        factor = (1 + contrast)
        img_array = (img_array - 0.5) * factor + 0.5
    
    img_array = np.clip(img_array, 0, 1)
    
    # Apply colormap using matplotlib
    cmap_name = settings['colormap']
    cmap = cm.get_cmap(cmap_name)
    
    # Apply colormap and convert to RGB (0-255)
    colored = cmap(img_array)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(colored_rgb)

# Initialize session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'background_image' not in st.session_state:
    st.session_state.background_image = None
if 'crop_coords' not in st.session_state:
    st.session_state.crop_coords = None
if 'time_settings' not in st.session_state:
    st.session_state.time_settings = {
        'start_time': 25.0,
        'interval': 25.0,
        'unit': 'µs'
    }

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 ICCD Image Processing Dashboard</h1>
    <p>Transform dark ICCD images into vibrant thermal visualizations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # File upload
    uploaded_files = st.file_uploader(
        "📤 Upload ICCD Images",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'tif'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Background image upload
    background_file = st.file_uploader(
        "🖼️ Background Image (Optional)",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'tif'],
        key="bg_uploader"
    )
    
    if background_file:
        st.session_state.background_image = Image.open(background_file)
        st.success("✅ Background loaded!")
    
    st.divider()
    
    # Crop Settings
    with st.expander("✂️ Crop Settings"):
        if st.session_state.uploaded_images:
            first_img = st.session_state.uploaded_images[0]['image']
            img_width, img_height = first_img.size
            
            col1, col2 = st.columns(2)
            with col1:
                crop_top = st.number_input("Top", 0, img_height, 0, 10)
                crop_left = st.number_input("Left", 0, img_width, 0, 10)
            with col2:
                crop_bottom = st.number_input("Bottom", 0, img_height, img_height, 10)
                crop_right = st.number_input("Right", 0, img_width, img_width, 10)
            
            if st.button("✅ Apply Crop", use_container_width=True):
                if crop_bottom > crop_top and crop_right > crop_left:
                    st.session_state.crop_coords = (crop_top, crop_bottom, crop_left, crop_right)
                    st.success(f"✂️ {crop_right-crop_left}×{crop_bottom-crop_top} px")
                else:
                    st.error("Invalid coordinates!")
            
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.crop_coords = None
        else:
            st.info("Upload images first")
    
    st.divider()
    
    # Normalization
    st.subheader("📊 Normalization")
    col1, col2 = st.columns(2)
    with col1:
        percentile_low = st.slider("Low %", 0, 10, 1)
    with col2:
        percentile_high = st.slider("High %", 90, 100, 99)
    
    st.divider()
    
    # Advanced Processing
    with st.expander("🔧 Advanced Filters"):
        use_bg_subtraction = st.checkbox(
            "Background Subtraction",
            value=False,
            disabled=st.session_state.background_image is None
        )
        
        use_clahe = st.checkbox("CLAHE Enhancement", value=False)
        if use_clahe:
            clahe_clip_limit = st.slider("CLAHE Clip", 0.01, 0.10, 0.03, 0.01)
        else:
            clahe_clip_limit = 0.03
        
        use_threshold = st.checkbox("Thresholding", value=False)
        if use_threshold:
            threshold_value = st.slider("Threshold", 0, 50, 5)
        else:
            threshold_value = 5
        
        use_smoothing = st.checkbox("Gaussian Smoothing", value=False)
        if use_smoothing:
            gaussian_sigma = st.slider("Sigma", 0.5, 5.0, 2.0, 0.5)
        else:
            gaussian_sigma = 2.0
    
    st.divider()
    
    # Colormap and adjustments
    colormap = st.selectbox(
        "🎨 Colormap",
        options=['inferno', 'plasma', 'viridis', 'hot', 'jet', 'magma', 'coolwarm'],
        index=0,
        help="Inferno works great for plasma jets!"
    )
    
    brightness = st.slider("💡 Brightness", 0, 200, 100, 1)
    contrast = st.slider("🔆 Contrast", 0, 200, 100, 1)
    
    st.divider()
    
    # Time label settings
    st.subheader("⏱️ Time Labels")
    col_time_1, col_time_2 = st.columns(2)
    with col_time_1:
        start_time = st.number_input(
            "Start Time",
            min_value=0.0,
            value=float(st.session_state.time_settings.get('start_time', 25.0)),
            step=1.0,
            format="%.3f"
        )
    with col_time_2:
        time_interval = st.number_input(
            "Interval",
            min_value=0.0,
            value=float(st.session_state.time_settings.get('interval', 25.0)),
            step=1.0,
            format="%.3f"
        )

    time_units = ["ns", "µs", "ms", "s"]
    default_unit = st.session_state.time_settings.get('unit', 'µs')
    try:
        default_index = time_units.index(default_unit)
    except ValueError:
        default_index = 1
    time_unit = st.selectbox("Time Unit", options=time_units, index=default_index)

    st.session_state.time_settings = {
        'start_time': start_time,
        'interval': time_interval,
        'unit': time_unit
    }

    st.divider()

    # Layout
    grid_cols = st.selectbox("📐 Grid Columns", [2, 3, 4, 5], index=2)
    show_labels = st.checkbox("🏷️ Overlay Time Labels", value=True)
    
    st.divider()
    
    # Clear button
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.uploaded_images = []
        st.session_state.processed_images = []
        st.session_state.background_image = None
        st.session_state.crop_coords = None
        st.rerun()

# Process uploaded files
if uploaded_files:
    if len(uploaded_files) != len(st.session_state.uploaded_images):
        st.session_state.uploaded_images = []
        for file in uploaded_files:
            file.seek(0)
            img = Image.open(file)
            st.session_state.uploaded_images.append({
                'name': file.name,
                'image': img,
                'timestamp': ""
            })

    for idx, img_data in enumerate(st.session_state.uploaded_images):
        timestamp_value = st.session_state.time_settings['start_time'] + idx * st.session_state.time_settings['interval']
        timestamp_text = f"Time = {timestamp_value:g} {st.session_state.time_settings['unit']}"
        img_data['timestamp'] = timestamp_text

# Process images
if st.session_state.uploaded_images:
    settings = {
        'colormap': colormap,
        'brightness': brightness,
        'contrast': contrast,
        'percentile_low': percentile_low,
        'percentile_high': percentile_high,
        'use_bg_subtraction': use_bg_subtraction,
        'use_clahe': use_clahe,
        'clahe_clip_limit': clahe_clip_limit,
        'use_threshold': use_threshold,
        'threshold_value': threshold_value,
        'use_smoothing': use_smoothing,
        'gaussian_sigma': gaussian_sigma,
        'crop_coords': st.session_state.crop_coords
    }
    
    st.session_state.processed_images = []
    
    # Show processing status
    with st.spinner('Processing images...'):
        for img_data in st.session_state.uploaded_images:
            try:
                processed = process_iccd_image(
                    img_data['image'],
                    st.session_state.background_image,
                    settings
                )
                overlay_image = add_timestamp_overlay(processed, img_data['timestamp'])
                st.session_state.processed_images.append({
                    'name': img_data['name'],
                    'image': processed,
                    'overlay_image': overlay_image,
                    'timestamp': img_data['timestamp']
                })
            except Exception as e:
                st.error(f"Error processing {img_data['name']}: {str(e)}")

# Display metrics
if st.session_state.processed_images:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.processed_images)}</div>
            <div class="metric-label">Images Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{percentile_low}-{percentile_high}%</div>
            <div class="metric-label">Percentile Range</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        filters = sum([use_bg_subtraction, use_clahe, use_threshold, use_smoothing])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{filters}</div>
            <div class="metric-label">Active Filters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{colormap.upper()[:3]}</div>
            <div class="metric-label">Colormap</div>
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
                    
                    # Display image with optional overlay
                    display_image = img_data['overlay_image'] if show_labels else img_data['image']
                    st.image(display_image, use_container_width=True)

                    # Download button
                    buffered = io.BytesIO()
                    download_image = display_image if show_labels else img_data['image']
                    download_image.save(buffered, format="PNG")
                    st.download_button(
                        label="⬇️ Download",
                        data=buffered.getvalue(),
                        file_name=f"processed_{img_data['name']}",
                        mime="image/png",
                        key=f"download_{img_idx}",
                        use_container_width=True
                    )
else:
    # Empty state
    st.markdown("""
    <div style='text-align: center; padding: 4rem; background: rgba(255, 255, 255, 0.1); 
                backdrop-filter: blur(10px); border-radius: 1rem; border: 2px dashed rgba(255, 255, 255, 0.3);'>
        <h2 style='color: white; margin-bottom: 1rem;'>📤 No Images Loaded</h2>
        <p style='color: rgba(255, 255, 255, 0.7); font-size: 1.1rem;'>
            Upload your ICCD images to see them transformed with thermal colormaps
        </p>
        <p style='color: rgba(255, 255, 255, 0.5); font-size: 0.9rem; margin-top: 1rem;'>
            Supports: PNG, JPG, TIFF, BMP
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: rgba(255, 255, 255, 0.5); padding: 1rem;'>
    <small>ICCD Image Processing Dashboard • Powered by Streamlit</small>
</div>
""", unsafe_allow_html=True)
