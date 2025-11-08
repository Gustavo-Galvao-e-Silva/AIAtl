import io
import os
import json
from datetime import datetime

import requests
from PIL import Image
import streamlit as st

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
st.markdown("<p class='subtitle'>Snap a photo → Send to backend → Identify the material & product type</p>", unsafe_allow_html=True)
st.divider()

# -------------- Sidebar Controls --------------
with st.sidebar:
    st.header("Settings")
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
    st.caption("In production, host the backend with HTTPS for camera access across browsers.")

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
                or "Unknown"
            )
            material = (
                payload.get("predicted_material")
                or payload.get("material")
                or payload.get("label")
                or payload.get("class")
                or "Unknown"
            )
            conf = payload.get("confidence") or payload.get("score") or payload.get("probability")
            tips = payload.get("tips") or payload.get("recommendations")

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
        if demo_mode:
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