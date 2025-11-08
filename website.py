import io
import os
import json
from datetime import datetime

import requests
from PIL import Image
import streamlit as st

# Import model predictor
MODEL_AVAILABLE = False
try:
    from model_predictor import get_model, predict_image
    MODEL_AVAILABLE = True
except ImportError as e:
    # Only show warning in sidebar, not at top level
    pass
except Exception as e:
    # Only show error in sidebar, not at top level
    pass

# -------------- Page Setup --------------
st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="centered")

# Small CSS touch
st.markdown(
    """
    <style>
      .centered-title { text-align: center; margin-bottom: 0.25rem; }
      .subtitle { text-align: center; color: #6b7280; margin-top: 0; }
      .stButton>button { width: 100%; }
      .muted { color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='centered-title'>RecycleVision</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Snap a photo → AI identifies recyclable materials & categories</p>", unsafe_allow_html=True)
st.divider()

# -------------- Sidebar Controls --------------
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    if MODEL_AVAILABLE:
        use_local_model = st.toggle(
            "Use Local Model",
            value=True,
            help="Use the trained local model for predictions (recommended).",
        )
    else:
        use_local_model = False
        try:
            from model_predictor import get_model, predict_image
            # If we get here, model is actually available
            MODEL_AVAILABLE = True
            use_local_model = st.toggle(
                "Use Local Model",
                value=True,
                help="Use the trained local model for predictions (recommended).",
            )
        except Exception as e:
            st.warning(f"Local model not available: {str(e)[:50]}... Using backend API.")
    
    if not use_local_model:
        backend_url = st.text_input(
            "Backend API URL",
            value=os.environ.get("RECYCLEVISION_API", "http://localhost:8000/predict"),
            help="Endpoint that accepts a multipart form upload with an image.",
        )
        demo_mode = st.toggle(
            "Demo mode (simulate response)",
            value=False,
            help="Use this if your backend isn't ready yet.",
        )
    else:
        demo_mode = False
        st.info("✓ Using local trained model")
        # Try to load model info
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, "model")
            model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier.pth")
            if os.path.exists(model_path):
                st.caption(f"Model: {os.path.basename(model_path)}")
        except:
            pass
    
    st.caption("In production, host with HTTPS for camera access across browsers.")

# -------------- Main Layout --------------
left, right = st.columns([1, 1], gap="large")

# We’ll keep a simple stateful area for results
def render_results_box(payload: dict | None):
    """
    Render the results container. If payload is None, show placeholders.
    """
    with st.container(border=True):
        st.write("### Results")
        if payload is None:
            st.write("**Type of Product (predicted):** —")
            st.write("**Material:** —")
            st.write("**Confidence:** —")
            st.caption("Run an analysis to populate this area.")
        else:
            # Map across possible key names from your backend
            pred_type = (
                payload.get("product_type")
                or payload.get("predicted_type")
                or payload.get("type")
                or payload.get("predicted_class")
                or "Unknown"
            )
            material = (
                payload.get("predicted_material")
                or payload.get("material")
                or payload.get("label")
                or payload.get("class")
                or payload.get("predicted_class")
                or "Unknown"
            )
            conf = payload.get("confidence") or payload.get("score") or payload.get("probability")
            tips = payload.get("tips") or payload.get("recommendations")
            
            # Show top predictions if available
            top_predictions = payload.get("top_predictions", [])

            st.write(f"**Type of Product (predicted):** {pred_type}")
            st.write(f"**Material:** {material}")

            if conf is not None:
                try:
                    conf_val = float(conf)
                    conf_pct = conf_val * 100 if conf_val <= 1.0 else conf_val
                    st.write(f"**Confidence:** {conf_pct:.1f}%")
                except Exception:
                    st.write(f"**Confidence:** {conf}")
            else:
                st.write("**Confidence:** —")

            if tips:
                st.write("**Tips**")
                if isinstance(tips, list):
                    for t in tips:
                        st.write(f"• {t}")
                else:
                    st.write(f"• {tips}")
            
            # Show top predictions if available
            if top_predictions and len(top_predictions) > 1:
                st.write("**Other possibilities:**")
                for i, pred in enumerate(top_predictions[1:], 1):
                    pred_conf = pred.get("confidence", 0) * 100 if pred.get("confidence", 0) <= 1 else pred.get("confidence", 0)
                    st.write(f"  {i}. {pred.get('class', 'Unknown')}: {pred_conf:.1f}%")

            with st.expander("Raw response"):
                st.code(json.dumps(payload, indent=2), language="json")


with left:
    st.subheader("Webcam")
    img_file = st.camera_input("Show webcam and take a picture", label_visibility="collapsed")

    # Optional: Allow upload from file as a fallback
    with st.expander("Or upload a file instead"):
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        if uploaded:
            img_file = uploaded  # reuse the same variable

    # Show a preview if available
    preview_image = None
    image_bytes = None
    filename = "image.jpg"

    if img_file is not None:
        # Get raw bytes safely from both camera_input and uploader
        if hasattr(img_file, "getvalue"):
            image_bytes = img_file.getvalue()
            filename = getattr(img_file, "name", "image.jpg")
        else:
            # Fallback for rare cases
            raw = img_file.read()
            image_bytes = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
            filename = getattr(img_file, "name", "image.jpg")

        try:
            preview_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(preview_image, caption="Preview", use_container_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            preview_image = None
            image_bytes = None

    # Actions
    colA, colB = st.columns(2)
    analyze_clicked = colA.button("🔎 Analyze", type="primary", use_container_width=True)
    save_clicked = colB.button("💾 Save photo locally", use_container_width=True)

    # Save locally
    if save_clicked:
        if preview_image is None:
            st.warning("Please capture or upload an image first.")
        else:
            out_dir = "captures"
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"recyclevision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            preview_image.save(path, format="JPEG", quality=92)
            st.success(f"Saved to {path}")

with right:
    # Initially show placeholders
    results_placeholder = st.empty()
    results_placeholder.write(render_results_box(None))

# -------------- Submission Logic --------------
if analyze_clicked:
    # Validate
    if image_bytes is None:
        st.error("Please capture or upload an image before analyzing.")
    else:
        if use_local_model and MODEL_AVAILABLE:
            # -------- LOCAL MODEL PREDICTION --------
            try:
                with st.spinner("Analyzing with local model…"):
                    # Get model (will cache after first load)
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    model, device, class_names = get_model(script_dir)
                    
                    # Load and predict image
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    payload = predict_image(model, image, class_names, device)
                    
                    # Add recycling tips based on material
                    material = payload.get("predicted_class", "").lower()
                    tips = []
                    if "plastic" in material:
                        tips = [
                            "Check local recycling guidelines for plastic types.",
                            "Rinse containers before recycling.",
                            "Remove caps and labels if required."
                        ]
                    elif "glass" in material:
                        tips = [
                            "Rinse glass containers thoroughly.",
                            "Remove metal caps and labels.",
                            "Don't break glass - recycle whole."
                        ]
                    elif "metal" in material or "aluminum" in material:
                        tips = [
                            "Rinse metal containers.",
                            "Aluminum cans are highly recyclable.",
                            "Check if local program accepts metal."
                        ]
                    elif "paper" in material or "cardboard" in material:
                        tips = [
                            "Keep paper dry and clean.",
                            "Remove plastic wrap or tape.",
                            "Flatten cardboard boxes."
                        ]
                    elif "e-waste" in material or "electronic" in material:
                        tips = [
                            "E-waste requires special handling.",
                            "Find local e-waste recycling centers.",
                            "Don't throw electronics in regular trash."
                        ]
                    elif "organic" in material:
                        tips = [
                            "Compost organic materials if possible.",
                            "Check local composting programs.",
                            "Remove any non-organic materials."
                        ]
                    else:
                        tips = [
                            "Check local recycling guidelines.",
                            "When in doubt, check with your local waste management."
                        ]
                    
                    payload["tips"] = tips
                
                # Update the Results panel
                with right:
                    results_placeholder.empty()
                    render_results_box(payload)
                st.success("Analysis complete!")
                
            except FileNotFoundError as e:
                st.error(f"Model not found: {e}")
                st.info("Please train the model first using main.py")
                with right:
                    results_placeholder.empty()
                    render_results_box({"error": "Model not found", "message": str(e)})
            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback
                with right:
                    results_placeholder.empty()
                    render_results_box({"error": "Prediction failed", "message": str(e)})
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        
        elif demo_mode:
            # -------- DEMO RESPONSE (no backend needed) --------
            with st.spinner("Analyzing (demo)…"):
                import random, time
                time.sleep(1.2)
                demo_types = ["Bottle", "Can", "Cup", "Jar", "Paper", "Cardboard", "Plastic bag/film", "Container"]
                demo_materials = ["Plastic (PET)", "Aluminum", "Glass", "Paper", "Cardboard", "Mixed/Unknown"]
                payload = {
                    "product_type": random.choice(demo_types),
                    "predicted_material": random.choice(demo_materials),
                    "confidence": round(random.uniform(0.78, 0.97), 2),
                    "tips": [
                        "Rinse if possible.",
                        "Remove caps/labels if required by local rules.",
                    ],
                }
            # Update the Results panel
            with right:
                results_placeholder.empty()
                render_results_box(payload)

        else:
            # -------- REAL BACKEND CALL --------
            if not backend_url.strip():
                st.error("Please set a valid Backend API URL in the sidebar.")
            else:
                try:
                    with st.spinner("Analyzing…"):
                        files = {"file": (filename, image_bytes, "image/jpeg")}
                        # No type is sent ahead of time now; just the image (adjust if your API needs more form fields)
                        r = requests.post(backend_url, files=files, timeout=30)

                    if r.status_code == 200:
                        try:
                            payload = r.json()
                        except Exception:
                            payload = {"raw_text": r.text}

                        with right:
                            results_placeholder.empty()
                            render_results_box(payload)
                        st.success("Analysis complete")
                    else:
                        st.error(f"Backend returned {r.status_code}")
                        with right:
                            results_placeholder.empty()
                            render_results_box({"error": f"HTTP {r.status_code}", "raw_text": r.text})
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
                    st.info("Check that your backend URL is reachable and CORS/HTTPS are configured as needed.")