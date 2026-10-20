import torch
import streamlit as st
from PIL import Image

from src.models import create_resnet18
from src.utils import get_val_transform


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bird Species Classification",
    page_icon="🐦",
    layout="wide"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = "resnet18_3stage_best.pth"
NUM_CLASSES = 200

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = create_resnet18(
        num_classes=NUM_CLASSES
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    model.eval()

    return model


model = load_model()


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = get_val_transform()


# ---------------------------------------------------------
# Class names
# ---------------------------------------------------------
#
# Replace these later with actual bird species names
# if you have a class-name mapping.
# ---------------------------------------------------------

class_names = [
    f"Class {i}"
    for i in range(NUM_CLASSES)
]


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_bird(image):
    """
    Predict the Top-5 bird classes for an uploaded image.
    """

    image = image.convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top5_probabilities, top5_indices = torch.topk(
            probabilities,
            k=5,
            dim=1
        )

    top5_probabilities = (
        top5_probabilities[0]
        .cpu()
        .tolist()
    )

    top5_indices = (
        top5_indices[0]
        .cpu()
        .tolist()
    )

    results = []

    for class_index, probability in zip(
        top5_indices,
        top5_probabilities
    ):

        results.append(
            {
                "class_name":
                    class_names[class_index],

                "class_id":
                    class_index,

                "confidence":
                    probability
            }
        )

    return results


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title(
    "🐦 Bird Species Classification"
)

st.write(
    """
    Upload an image of a bird and the trained **ResNet18**
    deep learning model will predict the most likely class.

    The application displays the **Top-5 predictions**
    together with confidence scores.
    """
)


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

with st.expander(
    "Model Information"
):

    st.write(
        """
        **Model:** ResNet18 with 3-stage progressive fine-tuning

        **Number of classes:** 200

        **Test Top-1 Accuracy:** 67.03%

        **Test Average Accuracy Per Class:** 66.32%

        **Test Top-5 Accuracy:** 87.54%

        The model uses transfer learning and progressive fine-tuning.
        Stage 1 trained the classifier only, Stage 2 fine-tuned layer4
        and the classifier, and Stage 3 fine-tuned layer3, layer4,
        and the classifier.
        """
    )


# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------

left_column, right_column = st.columns(
    [1, 1]
)


# ---------------------------------------------------------
# Image uploader
# ---------------------------------------------------------

with left_column:

    st.subheader(
        "Upload Bird Image"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded bird image",
            use_container_width=True
        )


# ---------------------------------------------------------
# Prediction output
# ---------------------------------------------------------

with right_column:

    st.subheader(
        "Top-5 Predictions"
    )

    if uploaded_file is None:

        st.info(
            "Upload a bird image to begin classification."
        )

    else:

        if st.button(
            "Classify Bird",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                predictions = predict_bird(
                    image
                )

            # -------------------------------------------------
            # Best prediction
            # -------------------------------------------------

            best_prediction = predictions[0]

            st.success(
                f"Prediction: "
                f"{best_prediction['class_name']}"
            )

            st.metric(
                label="Top-1 Confidence",
                value=(
                    f"{best_prediction['confidence'] * 100:.2f}%"
                )
            )

            st.divider()

            # -------------------------------------------------
            # Top-5 results
            # -------------------------------------------------

            for rank, prediction in enumerate(
                predictions,
                start=1
            ):

                confidence_percent = (
                    prediction["confidence"]
                    * 100
                )

                st.write(
                    f"**{rank}. "
                    f"{prediction['class_name']}**"
                )

                st.progress(
                    float(
                        prediction["confidence"]
                    )
                )

                st.caption(
                    f"Confidence: "
                    f"{confidence_percent:.2f}%"
                )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    """
    Bird Species Classification using a fine-tuned ResNet18
    deep learning model.
    """
)
