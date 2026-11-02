import streamlit as st
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Satellite Change Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Change Detection")
st.write(
    "Upload two satellite images of the same location "
    "from different times to detect changes."
)


# --------------------------------------------------
# Model setup
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from model import ChangeDetectionModel

MODEL_PATH = BASE_DIR / "models" / "phase4b_longer_training_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():

    model = ChangeDetectionModel().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def preprocess_image(image):

    image = image.convert("RGB")
    tensor = image_transform(image)
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


# --------------------------------------------------
# Upload images
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Image A — Before")

    uploaded_a = st.file_uploader(
        "Upload earlier satellite image",
        type=["png", "jpg", "jpeg"],
        key="image_a"
    )

with col2:

    st.subheader("Image B — After")

    uploaded_b = st.file_uploader(
        "Upload later satellite image",
        type=["png", "jpg", "jpeg"],
        key="image_b"
    )


# --------------------------------------------------
# Run prediction
# --------------------------------------------------

if uploaded_a is not None and uploaded_b is not None:

    image_a = Image.open(uploaded_a).convert("RGB")
    image_b = Image.open(uploaded_b).convert("RGB")

    display_col1, display_col2 = st.columns(2)

    with display_col1:
        st.image(image_a, caption="Image A — Before", use_container_width=True)

    with display_col2:
        st.image(image_b, caption="Image B — After", use_container_width=True)

    if st.button("Detect Changes", type="primary"):

        if not MODEL_PATH.exists():
            st.error(f"Model checkpoint not found: {MODEL_PATH}")
            st.stop()

        with st.spinner("Analyzing satellite images..."):

            model = load_model()

            tensor_a = preprocess_image(image_a)
            tensor_b = preprocess_image(image_b)

            with torch.no_grad():

                output = model(tensor_a, tensor_b)

                probabilities = torch.sigmoid(output)

                threshold = 0.10

                mask = (probabilities > threshold).float()

        mask_array = mask.squeeze().cpu().numpy()
        probability_array = probabilities.squeeze().cpu().numpy()

        # Binary mask: white = predicted change
        mask_image = (mask_array * 255).astype(np.uint8)

        # Resize images to model resolution for visualization
        image_a_resized = image_a.resize((256, 256))
        image_b_resized = image_b.resize((256, 256))

        # Overlay detected changes in red on Image B
        overlay = np.array(image_b_resized).copy()

        change_pixels = mask_array == 1

        overlay[change_pixels] = (
            0.5 * overlay[change_pixels] +
            0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        overlay_image = Image.fromarray(overlay)

        # Calculate predicted change percentage
        changed_percentage = mask_array.mean() * 100

        st.success("Change detection completed!")

        st.subheader("Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.image(
                image_a_resized,
                caption="Before",
                use_container_width=True
            )

        with result_col2:
            st.image(
                image_b_resized,
                caption="After",
                use_container_width=True
            )

        with result_col3:
            st.image(
                mask_image,
                caption="Predicted Change Mask",
                use_container_width=True
            )

        st.subheader("Change Overlay")

        st.image(
            overlay_image,
            caption="Red areas indicate predicted changes",
            use_container_width=True
        )

        st.metric(
            "Predicted changed area",
            f"{changed_percentage:.2f}%"
        )

        st.caption(
            "This percentage represents pixels classified as changed "
            "by the model, not a verified real-world area measurement."
        )

        st.caption(
            f"Model threshold: {threshold:.2f}"
        )

else:

    st.info("Upload both satellite images to begin.")