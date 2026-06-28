# SE01-PR-2026-Phase-II-Product-Image-classification-Ashutosh-Patil-
Product Image Classification Project Using CNN models.

---
title: E-commerce Product Classifier
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# E-commerce Product Classifier with Grad-CAM

**Live Product Image Classification using Deep Learning**

Production-ready Flask web application for e-commerce product classification using trained Custom CNN, MobileNetV2, and ResNet50 models with real-time Grad-CAM explainability visualization.

## ✨ Features

- **3 Trained Models**: Custom CNN, MobileNetV2, ResNet50 (all optimized)
- **Grad-CAM Visualization**: See exactly which image regions influenced predictions
- **Real-time Predictions**: Upload any image and get instant results
- **9 Product Categories**: BABY_PRODUCTS, BEAUTY_HEALTH, CLOTHING_ACCESSORIES_JEWELLERY, ELECTRONICS, GROCERY, HOBBY_ARTS_STATIONERY, HOME_KITCHEN_TOOLS, PET_SUPPLIES, SPORTS_OUTDOOR
- **Model Comparison**: Side-by-side metrics of all 3 models
- **Production-Grade**: Thread-safe, auto-cleanup, error handling, logging
- **Responsive UI**: Works perfectly on desktop, tablet, mobile
- **Accessible**: WCAG 2.1 compliant (keyboard navigation, screen readers)

## 🚀 How to Use

1. **Upload an Image**: Drag-and-drop or click to select a product image
2. **Select Models**: Choose which models to run (or run all 3)
3. **Get Predictions**: See confidence scores and Grad-CAM heatmaps
4. **Analyze Results**: View model comparisons and explanations

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Size |
|-------|----------|-----------|--------|----------|------|
| Custom CNN | 45.47% | 41.17% | 45.47% | 0.3858 | 8.9 MB |
| MobileNetV2 | 71.76% | 71.13% | 71.76% | 0.7106 | 33 MB |
| ResNet50 | 76.93% | 77.39% | 76.93% | 0.7680 | 333 MB |

## 🔧 Technical Details

- **Framework**: Flask (Python backend)
- **Models**: TensorFlow/Keras (.keras format)
- **Input Size**: 224×224 pixels
- **Classes**: 9 product categories
- **Explainability**: Grad-CAM overlay visualization
- **Deployment**: Docker on Hugging Face Spaces

## 📁 Structure

```
.
├── app.py                 # Flask backend (production-optimized)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── README.md             # This file
├── templates/
│   └── index.html        # HTML template
├── static/
│   ├── css/styles.css    # Styling (responsive design)
│   └── js/app.js         # Frontend (retry logic, state mgmt)
└── nn_ecommerce_outputs/
    ├── models/
    │   ├── custom_cnn.keras
    │   ├── mobilenetv2.keras
    │   └── resnet50.keras
    ├── metadata/
    │   ├── class_names.json
    │   └── model_manifest.json
    └── tables/
        └── (CSV files with metrics)
```

## 🛠 Production Features

✅ **Thread-safe model caching** - Safe for concurrent requests  
✅ **Automatic cleanup** - Old generated files cleaned up  
✅ **Retry logic** - Network failures handled gracefully  
✅ **Input validation** - File type & size checks  
✅ **Structured logging** - Debug everything  
✅ **Error handling** - User-friendly messages  
✅ **Progress tracking** - Real-time prediction progress  
✅ **Accessible UI** - WCAG 2.1 compliant  

## 📱 Browser Support

- Chrome/Chromium (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## 🚨 Troubleshooting

**Q: Models not loading?**
A: Check that `.keras` files exist in `nn_ecommerce_outputs/models/`

**Q: Predictions are slow?**
A: First prediction loads models (normal), subsequent are faster

**Q: Image upload fails?**
A: Check file size < 12 MB and format (JPG, PNG, WEBP, BMP)

**Q: Want to see logs?**
A: Check Space Settings → Logs tab for detailed information

## 📚 Resources

- [Flask Documentation](https://flask.palletsprojects.com)
- [TensorFlow/Keras](https://tensorflow.org)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [HF Spaces Docs](https://huggingface.co/docs/hub/spaces)

## 📄 License

MIT License - Free for academic and commercial use

## 🤝 About This Project

This is a comprehensive deep learning project for e-commerce product classification featuring:
- Custom CNN trained from scratch
- Transfer learning with MobileNetV2 and ResNet50
- Extensive explainability analysis with Grad-CAM
- Production-ready web deployment

**Author**: Ashutosh Rajendra Patil  
**Institution**: University of Europe for Applied Sciences  
**Dataset**: Kaggle ecommerce_product_images_18K (18,175 images, 9 categories)

---

**Status**: ✅ Production Ready | **Python**: 3.9+ | **TensorFlow**: 2.15.0 | **Updated**: June 2026



