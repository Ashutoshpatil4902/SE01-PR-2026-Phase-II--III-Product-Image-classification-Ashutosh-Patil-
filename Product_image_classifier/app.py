from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from threading import RLock

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AppConfig:
    """Production-grade configuration"""
    MAX_UPLOAD_SIZE = 30 * 1024 * 1024  # 30 MB
    PREDICTION_TIMEOUT = 120  # 2 minutes
    GENERATED_FILES_RETENTION_DAYS = 7
    GRADCAM_OVERLAY_BASE_WEIGHT = 0.58
    GRADCAM_OVERLAY_HEATMAP_WEIGHT = 0.42
    MODEL_CACHE_SIZE = 10

# ============================================================================
# PATHS & INITIALIZATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
KAGGLE_OUTPUT_DIR = Path("/kaggle/working/nn_ecommerce_outputs")

# Look for the artifact folder next to app.py first (the normal layout),
# then one level up, then the Kaggle path. First match wins.
def _find_default_artifact_dir():
    candidates = [
        BASE_DIR / "nn_ecommerce_outputs",
        PROJECT_DIR / "nn_ecommerce_outputs",
        KAGGLE_OUTPUT_DIR,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return BASE_DIR / "nn_ecommerce_outputs"

DEFAULT_ARTIFACT_DIR = _find_default_artifact_dir()

ARTIFACT_DIR = Path(os.getenv("NN_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)).resolve()
MODEL_DIR = Path(os.getenv("NN_MODEL_DIR", ARTIFACT_DIR / "models")).resolve()
METADATA_DIR = Path(os.getenv("NN_METADATA_DIR", ARTIFACT_DIR / "metadata")).resolve()
TABLE_DIR = Path(os.getenv("NN_TABLE_DIR", ARTIFACT_DIR / "tables")).resolve()
FIGURE_DIR = Path(os.getenv("NN_FIGURE_DIR", ARTIFACT_DIR / "figures")).resolve()
GENERATED_DIR = BASE_DIR / "static" / "generated"
DATASET_IMAGE_DIR = ARTIFACT_DIR / "dataset_images"
DATASET_IMAGE_MANIFEST_PATH = TABLE_DIR / "dataset_images_manifest.csv"
LOCAL_DATASET_DIR = PROJECT_DIR / "ECOMMERCE_PRODUCT_IMAGES"

GENERATED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ============================================================================
# THREAD-SAFE CACHING
# ============================================================================

_cache_lock = RLock()
_model_cache = {}
_dataset_cache = None
_dataset_cache_time = None
_tf_module = None
_keras_module = None

CACHE_TTL_SECONDS = 3600

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = AppConfig.MAX_UPLOAD_SIZE
app.config["JSON_SORT_KEYS"] = False

# ============================================================================
# UTILITIES
# ============================================================================

def validate_file_extension(filename, allowed=ALLOWED_EXTENSIONS):
    """Validate file extension"""
    if not filename:
        return False
    ext = Path(secure_filename(filename)).suffix.lower()
    return ext in allowed

def cleanup_old_generated_files():
    """Remove generated files older than retention period"""
    if not GENERATED_DIR.exists():
        return 0
    
    cutoff = datetime.now() - timedelta(days=AppConfig.GENERATED_FILES_RETENTION_DAYS)
    removed = 0
    
    try:
        for file in GENERATED_DIR.glob("*.png"):
            try:
                if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                    file.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {e}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
    
    return removed

def load_json(path, fallback):
    """Safely load JSON file"""
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
    return fallback

def normalize_artifact_path(value, allowed_extensions=None):
    """Normalize and validate artifact path (security)"""
    artifact_path = str(value or "").replace("\\", "/").lstrip("/")
    parts = Path(artifact_path).parts
    
    # Block dangerous patterns
    if not artifact_path or ".." in parts or artifact_path.startswith("/"):
        return None
    
    # Validate extension if provided
    if allowed_extensions:
        ext = Path(artifact_path).suffix.lower()
        if ext not in allowed_extensions:
            return None
    
    # Resolve and check symlinks
    try:
        file_path = (ARTIFACT_DIR / artifact_path).resolve()
        artifact_root = ARTIFACT_DIR.resolve()
        if not str(file_path).startswith(str(artifact_root)):
            return None
    except Exception:
        return None
    
    return artifact_path

def artifact_path_to_file(artifact_path):
    """Convert normalized path to file with security checks"""
    normalized = normalize_artifact_path(artifact_path)
    if not normalized:
        raise ValueError("Invalid artifact image path.")
    
    file_path = (ARTIFACT_DIR / normalized).resolve()
    artifact_root = ARTIFACT_DIR.resolve()
    if file_path != artifact_root and artifact_root not in file_path.parents:
        raise ValueError("Artifact image path is outside the output folder.")
    return file_path

# ============================================================================
# METADATA LOADERS
# ============================================================================

def class_names():
    """Load class names with caching"""
    names = load_json(METADATA_DIR / "class_names.json", [])
    return names if isinstance(names, list) else []

def model_manifest():
    """Load model manifest with fallback"""
    manifest = load_json(METADATA_DIR / "model_manifest.json", {})
    models = manifest.get("models", {}) if isinstance(manifest, dict) else {}
    
    if not models and MODEL_DIR.exists():
        logger.info("Building model manifest from filesystem")
        for path in sorted(MODEL_DIR.glob("*.keras")):
            key = path.stem.lower().replace(" ", "_")
            models[key] = {
                "file": path.name,
                "safe_name": key,
                "display_name": path.stem.replace("_", " ").title(),
                "last_conv_layer": None,
            }
    
    return {
        "input_size": manifest.get("input_size", [224, 224]) if isinstance(manifest, dict) else [224, 224],
        "models": models,
    }

def read_csv_table(path, max_rows=12):
    """Read CSV table with error handling"""
    if not path.exists():
        return None
    
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(next(iter([reader]), []))[:max_rows]
            return {
                "columns": reader.fieldnames or [],
                "rows": rows,
                "url": url_for("output_file", filename=f"tables/{path.name}"),
            }
    except Exception as e:
        logger.warning(f"Failed to read table {path}: {e}")
        return None

# ============================================================================
# DATASET IMAGE MANAGEMENT
# ============================================================================

def dataset_images_manifest():
    """Load dataset images with caching"""
    global _dataset_cache, _dataset_cache_time
    
    # Return cached if still valid
    if _dataset_cache is not None and _dataset_cache_time is not None:
        if (datetime.now() - _dataset_cache_time).total_seconds() < CACHE_TTL_SECONDS:
            return _dataset_cache
    
    rows = []
    
    # Try manifest file first
    if DATASET_IMAGE_MANIFEST_PATH.exists():
        try:
            with DATASET_IMAGE_MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader):
                    artifact_path = normalize_artifact_path(row.get("artifact_path"))
                    if not artifact_path:
                        continue
                    file_path = ARTIFACT_DIR / artifact_path
                    if not file_path.exists():
                        continue
                    rows.append({
                        "id": str(row.get("id") or index),
                        "source": "artifact",
                        "label": row.get("label") or file_path.parent.name,
                        "label_id": row.get("label_id"),
                        "filename": row.get("filename") or file_path.name,
                        "artifact_path": artifact_path,
                    })
        except Exception as e:
            logger.warning(f"Failed to read manifest: {e}")
    
    # Fallback to artifact directory
    if not rows and DATASET_IMAGE_DIR.exists():
        try:
            image_files = sorted(
                path for path in DATASET_IMAGE_DIR.rglob("*") 
                if path.suffix.lower() in ALLOWED_EXTENSIONS
            )
            for index, file_path in enumerate(image_files):
                rows.append({
                    "id": str(index),
                    "source": "artifact",
                    "label": file_path.parent.name,
                    "label_id": None,
                    "filename": file_path.name,
                    "artifact_path": file_path.relative_to(ARTIFACT_DIR).as_posix(),
                })
        except Exception as e:
            logger.warning(f"Failed to scan artifact directory: {e}")
    
    # Fallback to local directory
    if not rows and LOCAL_DATASET_DIR.exists():
        try:
            image_files = sorted(
                path for path in LOCAL_DATASET_DIR.rglob("*") 
                if path.suffix.lower() in ALLOWED_EXTENSIONS
            )
            for index, file_path in enumerate(image_files):
                rows.append({
                    "id": str(index),
                    "source": "local",
                    "label": file_path.parent.name,
                    "label_id": None,
                    "filename": file_path.name,
                    "local_path": file_path.relative_to(LOCAL_DATASET_DIR).as_posix(),
                })
        except Exception as e:
            logger.warning(f"Failed to scan local directory: {e}")
    
    _dataset_cache = rows
    _dataset_cache_time = datetime.now()
    logger.info(f"Loaded {len(rows)} dataset images")
    return rows

def dataset_image_by_id(image_id):
    """Find dataset image by ID"""
    image_id = str(image_id)
    for row in dataset_images_manifest():
        if row["id"] == image_id:
            return row
    return None

def dataset_image_url(row):
    """Get URL for dataset image"""
    if row.get("source") == "local":
        return url_for("dataset_image_file", image_id=row["id"])
    return url_for("output_file", filename=row["artifact_path"])

def dataset_image_file_path(row):
    """Get file path for dataset image with validation"""
    if row.get("source") == "local":
        local_path = normalize_artifact_path(row.get("local_path"))
        if not local_path:
            raise ValueError("Invalid local dataset image path.")
        
        file_path = (LOCAL_DATASET_DIR / local_path).resolve()
        dataset_root = LOCAL_DATASET_DIR.resolve()
        if file_path != dataset_root and dataset_root not in file_path.parents:
            raise ValueError("Local dataset image path outside dataset folder.")
        return file_path
    
    return artifact_path_to_file(row["artifact_path"])

# ============================================================================
# TENSORFLOW & MODEL LOADING (THREAD-SAFE)
# ============================================================================

def tensorflow_modules():
    """Load TensorFlow with thread safety"""
    global _tf_module, _keras_module
    
    with _cache_lock:
        if _tf_module is not None and _keras_module is not None:
            return _tf_module, _keras_module
        
        try:
            logger.info("Loading TensorFlow...")
            import tensorflow as tf
            # The models were saved with standalone Keras 3 (their config refers
            # to keras.src.models.functional). Loading them through tensorflow.keras
            # fails with "Could not deserialize class 'Functional'". So prefer the
            # standalone keras package and fall back to tf.keras only if absent.
            try:
                import keras
                logger.info(f"Using standalone Keras {keras.__version__}")
            except ImportError:
                from tensorflow import keras
                logger.info("Using tensorflow.keras (standalone keras not found)")
            logger.info(f"TensorFlow {tf.__version__} loaded")
            # Models were trained with mixed_float16 precision (see Kaggle notebook).
            try:
                keras.mixed_precision.set_global_policy("mixed_float16")
                logger.info("Mixed precision policy set: mixed_float16")
            except Exception as policy_exc:
                logger.warning(f"Could not set mixed_float16 policy: {policy_exc}")
        except ImportError:
            logger.error("TensorFlow not installed")
            raise RuntimeError("TensorFlow not installed. Run: pip install tensorflow")
        except Exception as exc:
            logger.error(f"TensorFlow initialization failed: {exc}")
            raise RuntimeError(f"TensorFlow init failed: {exc}") from exc
        
        _tf_module = tf
        _keras_module = keras
        return _tf_module, _keras_module

def model_path_for(model_info):
    """Get model file path"""
    return MODEL_DIR / model_info["file"]

def load_model(model_key):
    """Load model with thread-safe caching"""
    with _cache_lock:
        if model_key in _model_cache:
            logger.debug(f"Using cached model: {model_key}")
            return _model_cache[model_key]
        
        manifest = model_manifest()
        models = manifest["models"]
        
        if model_key not in models:
            logger.error(f"Unknown model: {model_key}")
            raise KeyError(f"Unknown model: {model_key}")
        
        model_info = models[model_key]
        model_path = model_path_for(model_info)
        
        if not model_path.exists():
            logger.error(f"Model file missing: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            logger.info(f"Loading model: {model_key} from {model_path}")
            tf, keras = tensorflow_modules()
            # The transfer models (MobileNetV2, ResNet50) contain a
            # Lambda(preprocess_input) layer that was saved by name only, so the
            # actual function must be supplied via custom_objects. The Custom CNN
            # has no Lambda and needs nothing extra.
            custom_objects = {}
            try:
                if model_key == "mobilenetv2":
                    from keras.applications.mobilenet_v2 import preprocess_input as _pre
                    custom_objects["preprocess_input"] = _pre
                elif model_key == "resnet50":
                    from keras.applications.resnet50 import preprocess_input as _pre
                    custom_objects["preprocess_input"] = _pre
            except Exception as pre_exc:
                logger.warning(f"Could not import preprocess_input for {model_key}: {pre_exc}")
            # safe_mode=False is required for the Lambda layers; compile=False
            # skips the optimizer state we don't need for inference.
            try:
                model = keras.models.load_model(
                    str(model_path), safe_mode=False, compile=False,
                    custom_objects=custom_objects or None,
                )
            except TypeError:
                model = keras.models.load_model(
                    str(model_path), custom_objects=custom_objects or None,
                )
            logger.info(f"Model loaded: {model_key}")
            
            _model_cache[model_key] = (model, model_info)
            
            # Limit cache size
            if len(_model_cache) > AppConfig.MODEL_CACHE_SIZE:
                oldest = next(iter(_model_cache))
                del _model_cache[oldest]
                logger.debug(f"Removed oldest cached model: {oldest}")
            
            return model, model_info
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def prepare_image(file, input_size):
    """Prepare uploaded image for prediction"""
    try:
        image = Image.open(file).convert("RGB")
        original_size = image.size
        
        image = ImageOps.fit(image, input_size, Image.Resampling.LANCZOS)
        # Models contain their own Rescaling / preprocess_input layers,
        # so they expect raw 0-255 float pixels (matches the Kaggle notebook).
        image_array = np.array(image, dtype="float32")
        
        # Save temp preview
        preview_name = f"preview_{uuid.uuid4().hex}.png"
        preview_path = GENERATED_DIR / preview_name
        image.save(preview_path)
        
        logger.info(f"Image processed: {original_size} -> {input_size}")
        return image_array, url_for("static", filename=f"generated/{preview_name}")
    except Exception as e:
        logger.error(f"Image preparation failed: {e}")
        raise ValueError(f"Invalid image file: {e}")

def prepare_artifact_image(image_path, input_size, image_url):
    """Prepare artifact image for prediction"""
    try:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, input_size, Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype="float32")
        return image_array, image_url
    except Exception as e:
        logger.error(f"Artifact image preparation failed: {e}")
        raise ValueError(f"Failed to load image: {e}")

# ============================================================================
# GRAD-CAM & PREDICTIONS
# ============================================================================

def colorize_heatmap(heatmap):
    """Convert grayscale heatmap to color"""
    try:
        import matplotlib
        try:
            cmap = matplotlib.colormaps["jet"]          # matplotlib >= 3.7
        except AttributeError:
            import matplotlib.cm as cm
            cmap = cm.get_cmap("jet")                    # older matplotlib
        return cmap(heatmap)[:, :, :3]
    except Exception:
        # Fallback: manual red colorization
        colored = np.zeros((*heatmap.shape, 3))
        colored[:, :, 0] = heatmap  # Red channel
        return colored

def _find_last_4d_layer_name(model):
    """Last layer whose output is 4D (B,H,W,C) — matches the notebook."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None

def make_gradcam_heatmap(model, img_array, pred_index, preferred_layer_name=None):
    """Generate Grad-CAM heatmap"""
    try:
        tf, keras = tensorflow_modules()
        
        # Prefer the manifest's layer if it actually exists in this model,
        # otherwise auto-detect the last 4D feature layer (notebook behavior).
        last_conv_layer_name = None
        if preferred_layer_name:
            try:
                model.get_layer(preferred_layer_name)
                last_conv_layer_name = preferred_layer_name
            except Exception:
                logger.info(f"Manifest layer '{preferred_layer_name}' not found; auto-detecting.")
        if not last_conv_layer_name:
            last_conv_layer_name = _find_last_4d_layer_name(model)
        
        if not last_conv_layer_name:
            logger.warning("No 4D feature layer found for Grad-CAM")
            return None, None
        
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = keras.models.Model(
            model.inputs, [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(img_array, axis=0), training=False)
            predictions = tf.cast(predictions, tf.float32)
            loss = predictions[:, pred_index]
        
        grads = tape.gradient(loss, conv_outputs)
        # Under mixed_float16 these can be float16; cast for stable math.
        conv_outputs = tf.cast(conv_outputs, tf.float32)
        grads = tf.cast(grads, tf.float32)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap)
        heatmap /= tf.math.reduce_max(heatmap) + 1e-10
        
        return heatmap.numpy(), last_conv_layer_name
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        return None, None

def save_gradcam_overlay(image_array, heatmap, model_key):
    """Save Grad-CAM overlay visualization"""
    try:
        heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
            (image_array.shape[1], image_array.shape[0]),
            Image.Resampling.BILINEAR,
        )
        heatmap_resized = np.asarray(heatmap_image).astype("float32") / 255.0
        colored_heatmap = colorize_heatmap(heatmap_resized)
        # image_array is raw 0-255; scale to 0-1 only for display (matches notebook)
        base = np.clip(image_array / 255.0, 0, 1)
        overlay = np.clip(
            (AppConfig.GRADCAM_OVERLAY_BASE_WEIGHT * base) + 
            (AppConfig.GRADCAM_OVERLAY_HEATMAP_WEIGHT * colored_heatmap),
            0, 1
        )
        
        overlay_name = f"gradcam_{model_key}_{uuid.uuid4().hex}.png"
        overlay_path = GENERATED_DIR / overlay_name
        Image.fromarray(np.uint8(overlay * 255)).save(overlay_path)
        
        logger.debug(f"Grad-CAM saved: {overlay_name}")
        return url_for("static", filename=f"generated/{overlay_name}")
    except Exception as e:
        logger.error(f"Grad-CAM save failed: {e}")
        return None

def top_predictions(predictions, labels, top_n=5):
    """Get top N predictions"""
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    return [
        {
            "label": labels[idx] if idx < len(labels) else f"Unknown {idx}",
            "confidence": float(predictions[idx]),
            "confidence_pct": int(predictions[idx] * 100),
        }
        for idx in top_indices if predictions[idx] > 0
    ]

def shap_figure_url(model_key):
    """Return URL for the precomputed SHAP reference figure, if present."""
    candidate = FIGURE_DIR / f"shap_{model_key}.png"
    if candidate.exists():
        return url_for("figure_file", filename=candidate.name)
    return None

def predict_with_models(image_array, image_url, selected_models):
    """Run predictions with all selected models"""
    labels = class_names()
    manifest = model_manifest()
    
    if not selected_models:
        selected_models = list(manifest["models"].keys())
    
    results = []
    for model_key in selected_models:
        try:
            logger.info(f"Running inference: {model_key}")
            model, model_info = load_model(model_key)
            predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
            top_rows = top_predictions(predictions, labels)
            best_index = int(np.argmax(predictions))
            
            heatmap, layer_name = make_gradcam_heatmap(
                model, image_array, best_index,
                preferred_layer_name=model_info.get("last_conv_layer"),
            )
            heatmap_url = save_gradcam_overlay(image_array, heatmap, model_key) if heatmap is not None else None
            
            results.append({
                "model_key": model_key,
                "display_name": model_info.get("display_name", model_key),
                "top_label": top_rows[0]["label"],
                "confidence": top_rows[0]["confidence"],
                "confidence_pct": top_rows[0]["confidence_pct"],
                "top_predictions": top_rows,
                "heatmap_url": heatmap_url,
                "gradcam_layer": layer_name,
                "model_file": model_info.get("file"),
                "error": None,
            })
            logger.info(f"Inference complete: {model_key} -> {top_rows[0]['label']}")
        
        except KeyError as e:
            logger.warning(f"Model config error {model_key}: {e}")
            results.append({
                "model_key": model_key,
                "display_name": manifest["models"].get(model_key, {}).get("display_name", model_key),
                "error": f"Model not found: {model_key}",
            })
        except FileNotFoundError as e:
            logger.warning(f"Model file missing {model_key}: {e}")
            results.append({
                "model_key": model_key,
                "display_name": manifest["models"].get(model_key, {}).get("display_name", model_key),
                "error": "Model file missing. Check server configuration.",
            })
        except Exception as e:
            logger.exception(f"Unexpected error in {model_key}")
            results.append({
                "model_key": model_key,
                "display_name": manifest["models"].get(model_key, {}).get("display_name", model_key),
                "error": f"Inference failed: {type(e).__name__}: {e}",
            })
    
    return {"image_url": image_url, "results": results}

# ============================================================================
# APP STATUS
# ============================================================================

def app_status():
    """Get application status"""
    labels = class_names()
    manifest = model_manifest()
    dataset_images = dataset_images_manifest()
    models = []
    
    for key, info in manifest["models"].items():
        path = model_path_for(info)
        models.append({
            "key": key,
            "display_name": info.get("display_name", key),
            "file": info.get("file"),
            "exists": path.exists(),
            "last_conv_layer": info.get("last_conv_layer"),
            "total_parameters": info.get("total_parameters"),
        })
    
    tensorflow_available = True
    tensorflow_message = "Ready"
    try:
        tensorflow_modules()
    except Exception as exc:
        tensorflow_available = False
        tensorflow_message = str(exc)
        logger.warning(f"TensorFlow unavailable: {exc}")
    
    comparison_path = TABLE_DIR / "final_comparison_table.csv"
    metrics_path = TABLE_DIR / "model_metrics.csv"
    report_path = ARTIFACT_DIR / "ecommerce_nn_explainability_report.pdf"
    
    ready = bool(labels) and any(model["exists"] for model in models) and tensorflow_available
    
    return {
        "ready": ready,
        "artifact_dir": str(ARTIFACT_DIR),
        "model_dir": str(MODEL_DIR),
        "metadata_dir": str(METADATA_DIR),
        "class_count": len(labels),
        "models": models,
        "tensorflow_available": tensorflow_available,
        "tensorflow_message": tensorflow_message,
        "tables": {
            "final_comparison": read_csv_table(comparison_path),
            "metrics": read_csv_table(metrics_path),
        },
        "dataset_images": {
            "count": len(dataset_images),
            "classes": sorted({row["label"] for row in dataset_images}),
            "dir": str(DATASET_IMAGE_DIR),
            "local_dir": str(LOCAL_DATASET_DIR),
            "manifest": str(DATASET_IMAGE_MANIFEST_PATH),
        },
        "report_url": url_for("output_file", filename="ecommerce_nn_explainability_report.pdf")
        if report_path.exists() else None,
    }

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    """Serve main HTML"""
    return render_template("index.html")

@app.route("/api/status")
def status():
    """Get app status"""
    return jsonify(app_status())

@app.route("/api/dataset-images")
def dataset_images():
    """Get paginated dataset images"""
    try:
        limit = max(1, min(int(request.args.get("limit", 24)), 96))
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        limit, offset = 24, 0
    
    label_filter = (request.args.get("label") or "").strip()
    all_images = dataset_images_manifest()
    labels = sorted({row["label"] for row in all_images})
    filtered = [row for row in all_images if not label_filter or row["label"] == label_filter]
    page_rows = filtered[offset : offset + limit]
    
    items = []
    for row in page_rows:
        item = dict(row)
        item["url"] = dataset_image_url(row)
        items.append(item)
    
    next_offset = offset + len(items) if offset + len(items) < len(filtered) else None
    
    return jsonify({
        "items": items,
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
        "next_offset": next_offset,
        "classes": labels,
    })

@app.route("/dataset-images/<image_id>")
def dataset_image_file(image_id):
    """Serve dataset image"""
    image_info = dataset_image_by_id(image_id)
    if image_info is None:
        return jsonify({"error": "Image not found"}), 404
    
    try:
        image_path = dataset_image_file_path(image_info)
    except Exception as exc:
        logger.error(f"Dataset image error: {exc}")
        return jsonify({"error": str(exc)}), 400
    
    return send_from_directory(image_path.parent, image_path.name)

@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict on uploaded image"""
    labels = class_names()
    if not labels:
        logger.error("class_names.json not found")
        return jsonify({"error": "Classifier not configured. Run Kaggle notebook first."}), 400
    
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify({"error": "No image uploaded."}), 400
    
    if not validate_file_extension(image_file.filename):
        return jsonify({"error": "Invalid image format. Use JPG, PNG, WEBP, or BMP."}), 400
    
    try:
        manifest = model_manifest()
        selected_models = request.form.getlist("models") or list(manifest["models"].keys())
        
        logger.info(f"Prediction request: {image_file.filename} with models {selected_models}")
        image_array, image_url = prepare_image(image_file, manifest["input_size"])
        return jsonify(predict_with_models(image_array, image_url, selected_models))
    except ValueError as e:
        logger.warning(f"Invalid image: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": "Prediction failed. Check server logs."}), 500

@app.route("/api/predict-dataset", methods=["POST"])
def predict_dataset():
    """Predict on dataset image"""
    labels = class_names()
    if not labels:
        return jsonify({"error": "Classifier not configured."}), 400
    
    payload = request.get_json(silent=True) or {}
    selected_models = payload.get("models") or []
    if isinstance(selected_models, str):
        selected_models = [selected_models]
    
    image_info = dataset_image_by_id(payload.get("image_id"))
    if image_info is None:
        return jsonify({"error": "Dataset image not found."}), 404
    
    try:
        manifest = model_manifest()
        image_url = dataset_image_url(image_info)
        image_path = dataset_image_file_path(image_info)
        
        logger.info(f"Dataset prediction: {image_info.get('filename')} with models {selected_models}")
        image_array, image_url = prepare_artifact_image(image_path, manifest["input_size"], image_url)
        return jsonify(predict_with_models(image_array, image_url, selected_models))
    except Exception as e:
        logger.exception("Dataset prediction error")
        return jsonify({"error": "Dataset prediction failed."}), 500

@app.route("/figures/<path:filename>")
def figure_file(filename):
    """Serve precomputed reference figures (SHAP, confusion matrices, etc.)"""
    safe = os.path.basename(filename)
    target = (FIGURE_DIR / safe).resolve()
    if not str(target).startswith(str(FIGURE_DIR.resolve())) or not target.exists():
        return jsonify({"error": "Figure not found"}), 404
    return send_from_directory(FIGURE_DIR, safe)

@app.route("/outputs/<path:filename>")
def output_file(filename):
    """Serve output files"""
    try:
        # Validate path to prevent traversal
        if not normalize_artifact_path(filename):
            return jsonify({"error": "Invalid file path"}), 400
        return send_from_directory(ARTIFACT_DIR, filename)
    except Exception as e:
        logger.warning(f"Output file error: {e}")
        return jsonify({"error": "File not found"}), 404

@app.before_request
def periodic_cleanup():
    """Clean old generated files (every 10 requests)"""
    if not hasattr(app, "request_count"):
        app.request_count = 0
    
    app.request_count += 1
    if app.request_count % 10 == 0:
        removed = cleanup_old_generated_files()
        if removed > 0:
            logger.info(f"Cleaned {removed} old files")

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large"""
    return jsonify({"error": "File too large. Max 30 MB."}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle server errors"""
    logger.exception("Internal server error")
    return jsonify({"error": "Server error. Check logs."}), 500

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    logger.info("=== Starting Ecommerce Product Classifier ===")
    logger.info(f"Artifact dir: {ARTIFACT_DIR}")
    logger.info(f"Model dir: {MODEL_DIR}")
    
    # Check TensorFlow at startup
    try:
        tensorflow_modules()
    except RuntimeError as e:
        logger.warning(f"TensorFlow not available at startup: {e}")
    
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
