import numpy as np
import streamlit as st
import supervision as sv
import yaml
import openai
import os
import io
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
BANNER_PATH = PROJECT_ROOT / "assets" / "Banner.png"
SAMPLE_IMAGES_DIR = Path(__file__).parent / "assets"
LOCAL_LABELS_WEIGHT = Path("/Users/andresetiawan/Training/AI/capstone-yolo/labels_weight")
if LOCAL_LABELS_WEIGHT.exists():
    LABELS_WEIGHT_DIR = LOCAL_LABELS_WEIGHT.resolve()
else:
    LABELS_WEIGHT_DIR = PROJECT_ROOT / "labels_weight"
if Path("/Users/andresetiawan/Training/AI/capstone-yolo/data.yaml").exists():
    DATA_YAML_PATH = Path("/Users/andresetiawan/Training/AI/capstone-yolo/data.yaml").resolve()
else:
    DATA_YAML_PATH = Path(__file__).parent.parent.parent / "data.yaml"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def load_model(model_name: str):
    model_path = MODELS_DIR / f"best_{model_name}.pt"
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()
    return YOLO(str(model_path))

@st.cache_resource
def get_annotators():
    return sv.BoxAnnotator(), sv.LabelAnnotator()

def detector_pipeline_pillow(image_bytes, model):
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np_rgb = np.array(pil_image)
    results = model(image_np_rgb, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()
    box_annotator, label_annotator = get_annotators()
    annotated_image = pil_image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image_np = np.asarray(annotated_image)
    class_names = detections.data.get("class_name", [])
    classcounts = dict(Counter(class_names))
    return annotated_image_np, classcounts

Calorie_DB_per_100g = {
    "potato": 77,
    "sausage": 250,
    "grilled_vegetables": 35,
    "cheese_sticks": 320,
    "water": 0,
    "rice": 130,
    "black_tea": 1,
    "rice_porridge": 50,
    "tiramisu": 240,
    "mayonnaise": 680,
    "broccoli": 34,
    "lemonade": 40,
}

Exercise_DB = {
    "running (30 minutes)": 300,
    "jogging (30 minutes)": 150,
    "cycling (30 minutes)": 250,
    "skipping (30 minutes)": 350
}

def parse_weight_label(label_file_path, class_names_dict):
    items = []
    if not label_file_path or not label_file_path.exists():
        return items
    with open(label_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            weight = float(parts[5])
            items.append({"class": class_names_dict[class_id], "weight": weight})
    return items

def calc_calories(class_name, weight_grams):
    kcal_per_100g = Calorie_DB_per_100g.get(class_name.lower(), 0)
    return kcal_per_100g * (weight_grams / 100)

def recommend_exercise(total_calories):
    recommendations = {}
    for ex, burn_kcal in Exercise_DB.items():
        duration_min = total_calories / burn_kcal * 30
        recommendations[ex] = round(duration_min, 1)
    return recommendations

with open(DATA_YAML_PATH) as f:
    data_yaml = yaml.safe_load(f)
class_names_dict = data_yaml['names']

def query_gpt4o_mini(items, total_calories):
    prompt_items = "\n".join([f"- {item['class']}: {item['weight']}g" for item in items])
    prompt = f"""
I have the following foods in their weights:
{prompt_items}

Total calories: {total_calories:.1f} kcal

Please provide:
1. A brief nutrition tip
2. Exercise recommendations to burn the total calories
Give the answer in concise plain text.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error LLM: {str(e)}"

def get_label_for_uploaded(uploaded_filename, labels_folder):
    uploaded_base = Path(uploaded_filename).stem
    labels_folder = Path(labels_folder)
    candidates = list(labels_folder.glob("*.txt"))
    for f in candidates:
        if uploaded_base in f.stem:
            return f
    return None

with st.sidebar:
    st.markdown("### Andre Setiawan")
    st.caption("Capstone Project Module 04")
    st.divider()
    st.markdown("**Application Function**")
    st.markdown("- Object Detection on Image")
    st.markdown("- Calorie Estimation")
    st.markdown("- Workout Recommendation")
    st.divider()
    st.caption("Built with Streamlit, YOLO & LLM")

st.image(str(BANNER_PATH), width=700)
st.title("YOLO-Based Food Portion Detection and Calorie Estimation System")

sample_files = list(SAMPLE_IMAGES_DIR.glob("*.jpg")) + list(SAMPLE_IMAGES_DIR.glob("*.png"))
if not sample_files:
    st.sidebar.write("No sample images found!")

sample_choice = st.sidebar.selectbox(
    "Choose a demo image",
    options=["None"] + [f.name for f in sample_files]
)

uploaded_file = None
if sample_choice != "None":
    sample_path = SAMPLE_IMAGES_DIR / sample_choice
    with open(sample_path, "rb") as f:
        bytes_data = f.read()
    uploaded_file = io.BytesIO(bytes_data)
    uploaded_file.name = sample_path.name

uploaded_file_manual = st.file_uploader(
    "Upload Image",
    accept_multiple_files=False,
    type=["jpg", "jpeg", "png", "webp"]
)
if uploaded_file_manual is not None:
    uploaded_file = uploaded_file_manual

with st.spinner("Loading model..."):
    model = load_model("food_portion_benchmark")

items = []
total_calories = 0

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=700)

    if st.button("Detect & Calculate", type="primary"):
        bytes_data = uploaded_file.getvalue()
        with st.spinner("Detecting objects..."):
            annotated_image_rgb, classcounts = detector_pipeline_pillow(bytes_data, model)

        st.subheader("Detection Results")
        st.image(annotated_image_rgb, caption="Detected Objects", width=700)

        if classcounts:
            st.subheader("Object Counts")
            col1, col2 = st.columns([1, 2])
            with col1:
                for class_name, count in classcounts.items():
                    st.metric(label=class_name, value=count)
        else:
            st.info("No objects detected in the image")

        uploaded_name = uploaded_file.name
        label_path = get_label_for_uploaded(uploaded_name, LABELS_WEIGHT_DIR)

        st.text(f"Uploaded file name: {uploaded_name}")
        st.text(f"Full label path: {label_path}")
        st.text(f"File exists? {label_path.exists() if label_path else False}")

        items = parse_weight_label(label_path, class_names_dict)
        st.write("Parsed items:", items)

        total_calories = sum(
            calc_calories(item["class"], item["weight"])
            for item in items
        )

        st.subheader("Calorie Breakdown")
        for item in items:
            cals = calc_calories(item["class"], item["weight"])
            st.write(f"{item['class']} ({item['weight']}g) → {cals:.1f} kcal")

        st.subheader("Total Calories")
        st.metric("Total Calories", f"{total_calories:.1f} kcal")

        st.subheader("Exercise Recommendations (basic)")
        recs = recommend_exercise(total_calories)
        for ex, dur in recs.items():
            st.write(f"{ex} → {dur} minutes")

        st.subheader("LLM Tips & Personalized Recommendations")
        with st.spinner("Querying GPT-4o Mini..."):
            llm_output = query_gpt4o_mini(items, total_calories)
            st.text(llm_output)