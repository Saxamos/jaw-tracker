from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim

from PIL import Image


def compute_euclidean_distance(image_1, image_2):
    return np.linalg.norm(image_1 - image_2)


def compute_ssim(image_1, image_2):
    image_2_resized = cv2.resize(image_2, (image_1.shape[1], image_1.shape[0]))
    gray1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image_2_resized, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score


IMAGE_EXT = [".jpg", ".jpeg", ".png"]

st.title("Welcome in Jaw Tracker 🦈")

uploaded_file = st.file_uploader(
    "Upload a panoramic radio 🦷", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    uploaded_image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    closest_image_path = None
    closest_distance = float("inf")

    folder_path = Path("local_db")
    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() in IMAGE_EXT:
            image = cv2.imread(file_path)
            distance = compute_ssim(image, uploaded_image_cv)
            print(f"distance with {file_path}: {distance}")
            if distance < closest_distance:
                closest_distance = distance
                closest_image_path = file_path

    st.image(
        Image.open(closest_image_path),
        caption=f"The closest found image in our DB is that one: {closest_image_path}",
        use_column_width=True,
    )

# collector = FeedbackCollector("", "", "")
# feedback = collector.st_feedback(
#     component="feedback", feedback_type="thumbs", model="distance"
# )
#
# # print out the feedback object as a dictionary in your app
# feedback.dict() if feedback else None
