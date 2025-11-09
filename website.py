import io
import os
import json
from datetime import datetime
from dotenv import load_dotenv

import requests
from PIL import Image
import streamlit as st

from openai import OpenAI

# Load the .env file
load_dotenv()

# Get the key safely
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found! Check your .env file location or spelling.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

MODEL_AVAILABLE = False
try: #importing our helper module for connecting to our model
    from model_predictor import get_model, predict_image
    MODEL_AVAILABLE = True
except ImportError:
    pass
except Exception:
    pass

# -------------- Page Setup --------------
st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="centered")

# CSS
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
st.markdown("<p class='subtitle'>Snap a photo → Find the recycling category for you! </p>", unsafe_allow_html=True)
st.divider()

# -------------- Sidebar --------------
with st.sidebar:
    st.header("Settings")

    # Model selection
    if MODEL_AVAILABLE:
        use_local_model = st.toggle(
            "Use Local Model",
            value=True,
            help="Use the trained local model for predictions (recommended).",
        )
        st.info("✓ Using local trained model")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, "model")
            model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier.pth")
            if os.path.exists(model_path):
                st.caption(f"Model: {os.path.basename(model_path)}")
        except:
            pass
    else:
        use_local_model = False
        backend_url = st.text_input(
            "Backend API URL",
            value=os.environ.get("RECYCLEVISION_API", "http://localhost:8000/predict"),
            help="Endpoint that accepts a multipart form upload with an image.",
        )
        st.warning("Local model not available — using backend API.")

    st.caption("For production, host with HTTPS for camera access across browsers.")

# -------------- Main Layout --------------
left, right = st.columns([1, 1], gap="large")

def render_results_box(payload: dict | None):
    with st.container(border=True):
        st.write("### Results")
        if payload is None:
            st.write("**Type of Product (predicted):** —")
            st.write("**Material:** —")
            st.write("**Confidence:** —")
            st.caption("Run an analysis to populate this area.")
        else:
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

    with st.expander("Or upload a file instead"):
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        if uploaded:
            img_file = uploaded

    preview_image = None
    image_bytes = None
    filename = "image.jpg"

    if img_file is not None:
        if hasattr(img_file, "getvalue"):
            image_bytes = img_file.getvalue()
            filename = getattr(img_file, "name", "image.jpg")
        else:
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

    colA, colB = st.columns(2)
    analyze_clicked = colA.button("🔎 Analyze", type="primary", use_container_width=True)
    save_clicked = colB.button("💾 Save photo locally", use_container_width=True)

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
    results_placeholder = st.empty()
    results_placeholder.write(render_results_box(None))

# -------------- Analysis Logic --------------
if analyze_clicked:
    if image_bytes is None:
        st.error("Please capture or upload an image before analyzing.")
    else:
        if use_local_model and MODEL_AVAILABLE:
            # -------- LOCAL MODEL PREDICTION --------
            try:
                with st.spinner("Analyzing with local model…"):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    model, device, class_names = get_model(script_dir)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    payload = predict_image(model, image, class_names, device)
                    
                    material = payload.get("predicted_class", "").lower()
                    print(material)
                    tips = []
                    if material:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",  # lightweight, fast model; good choice
                            messages=[
                                {"role": "system", "content": "You are a helpful recycling assistant."},
                                {"role": "user", "content": f"What are 3 tips for recycling objects made of {material}?"}
                            ],
                        )
                        tips = [response.choices[0].message.content]
                        
                    else:
                        tips = [
                            "Check local recycling guidelines.",
                            "When in doubt, check with your local waste management."
                        ]
                    payload["tips"] = tips

                with right:
                    results_placeholder.empty()
                    render_results_box(payload)
                st.success("Analysis complete!")

            except FileNotFoundError as e:
                st.error(f"Model not found: {e}")
                st.info("Please train the model first using main.py")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

        else:
            # -------- BACKEND PREDICTION --------
            if not MODEL_AVAILABLE and not backend_url.strip():
                st.error("Please set a valid Backend API URL in the sidebar.")
            else:
                try:
                    with st.spinner("Analyzing via backend API…"):
                        files = {"file": (filename, image_bytes, "image/jpeg")}
                        r = requests.post(backend_url, files=files, timeout=30)

                    if r.status_code == 200:
                        try:
                            payload = r.json()
                        except Exception:
                            payload = {"raw_text": r.text}

                        with right:
                            results_placeholder.empty()
                            render_results_box(payload)
                        st.success("Analysis complete!")
                    else:
                        st.error(f"Backend returned {r.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
