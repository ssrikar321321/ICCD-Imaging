# 🔬 ICCD Image Processing Dashboard

Advanced image processing dashboard for ICCD (Intensified Charge-Coupled Device) images with thermal colormaps, background subtraction, and noise reduction.

## Features

✨ **Advanced Processing:**
- Background subtraction for noise removal
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian smoothing for noise reduction
- Adjustable thresholding
- Multiple thermal colormaps (Jet, Inferno, Hot, Viridis)

🎨 **Interactive Controls:**
- Real-time image processing
- Brightness and contrast adjustment
- Customizable grid layout (2-5 columns)
- Batch processing support
- Individual image download

📊 **Perfect for:**
- Plasma physics research
- ICCD camera data analysis
- Time-sequence visualization
- Scientific image processing

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/iccd-dashboard.git
cd iccd-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run iccd_dashboard.py
```

### Usage

1. Upload your ICCD images
2. (Optional) Upload a background image for subtraction
3. Enable processing filters as needed
4. Adjust colormap and settings
5. Download processed images

## Processing Pipeline

```
Raw Image
    ↓
Background Subtraction (optional)
    ↓
CLAHE Enhancement (optional)
    ↓
Gaussian Smoothing (optional)
    ↓
Thresholding (optional)
    ↓
Brightness/Contrast Adjustment
    ↓
Colormap Application
    ↓
Final Result
```

## Technologies Used

- **Streamlit** - Interactive web interface
- **NumPy** - Array processing
- **Pillow** - Image handling
- **SciPy** - Gaussian filtering
- **scikit-image** - Advanced image processing (CLAHE)

## License

MIT License - feel free to use for research and commercial purposes

## Author

Your Name - Plasma Physics Research

## Acknowledgments

Built for ICCD image analysis in plasma jet research
