import streamlit as st
from PIL import Image
from io import BytesIO
import model

st.set_page_config(layout="wide", page_title="Object Detector")

st.write("## Detect objects in your image")
st.write(
    ":dog: Try uploading an image to watch the objects magically detected. " +
    "Full quality images can be downloaded from the sidebar. " + 
    "This code is open source and available " + 
    "[here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. "
)
st.sidebar.write("## Upload and download :gear:")

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload, model_name):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = Image.fromarray(get_detections(image, model_name))
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image",
                               convert_image(fixed), "fixed.png", "image/png")

def get_detections(image, model_name):
    use_KL_Loss = False 
    model_path = './model_w_without_kl.pth'
    if model_name == 'Faster RCNN + KL Loss':
        model_path = './model_w_with_kl_3.pth'
        use_KL_Loss = True
    net = model.get_model(use_KL_Loss, model_path = model_path)
    fixed_image = model.get_sample_prediction(net, image)
    return fixed_image

col1, col2 = st.columns(2)
my_model = st.sidebar.selectbox("Choose model", ['Faster RCNN', 'Faster RCNN + KL Loss'])
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload, model_name = my_model)
else:
    fix_image("./zebra.jpg", model_name = my_model)
