import os
from io import BytesIO

import streamlit as st
import wget
from PIL import Image

import model

model_url_dict = {
    'Faster-RCNN-Resnet50-FPN': 'https://github.com/souradipp76/box_regression-kl_loss/releases/download/v1.0/model_without_kl.pth', 
    'Faster-RCNN-Resnet50-FPN + KL Loss': 'https://github.com/souradipp76/box_regression-kl_loss/releases/download/v1.0/model_with_kl.pth'
}

model_path_dict = {
    'Faster-RCNN-Resnet50-FPN': './model_without_kl.pth', 
    'Faster-RCNN-Resnet50-FPN + KL Loss': './model_with_kl.pth'
}

model_conf_dict = {
    'Faster-RCNN-Resnet50-FPN': {}, 
    'Faster-RCNN-Resnet50-FPN + KL Loss': {'use_kl_loss': True}
}

st.set_page_config(layout="wide", page_title="Object Detector")

st.write("## Detect objects in your Image")
st.write(
    ":dog: Try uploading an image to watch the objects get magically detected. " +
    "Full quality images can be downloaded from the sidebar. " + 
    "Object detection models used here are only for experimental use. " +
    "The code is open source and available " + 
    "[here](https://github.com/souradipp76/object_detection_kl_app) on GitHub. "+
    "Check out more about this project [here](https://github.com/souradipp76/box_regression-kl_loss)."
)
st.sidebar.write("## Upload and Download :gear:")

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload, model_name):
    image = Image.open(upload).convert('RGB')
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = Image.fromarray(get_detections(image, model_name))
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image",
                               convert_image(fixed), "fixed.png", "image/png")

def get_detections(image, model_name):
    model_path = model_path_dict[model_name]
    cfg = model_conf_dict[model_name]
    model_url = model_url_dict[model_name]
    if not os.path.isfile(model_path):
        download_model(model_url, model_path)
    net = model.get_model(model_path = model_path, cfg=cfg)
    fixed_image = model.get_sample_prediction(net, image)
    return fixed_image

@st.cache()
def download_model(model_url, model_path):
    output = wget.download(model_url, model_path)
    return output


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
my_model = st.sidebar.selectbox("Choose Model", list(model_url_dict.keys()))


if my_upload is not None:
    fix_image(upload=my_upload, model_name = my_model)
else:
    fix_image("./voc.jpg", model_name = my_model)
