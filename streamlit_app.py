import streamlit as st 
import torch
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title 
st.title('Tomato Disease Classification')

#Set Header 
st.header('Please up load picture')


#Load Model 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# แก้ไขส่วนการโหลดโมเดล
# แทนที่จะใช้ torch.load โดยตรง ให้ใช้การโหลด state_dict แทน
# สมมติว่าคุณมีคลาสโมเดลเดิม (ถ้าไม่มีต้องนำเข้า)
from torchvision.models import mobilenet_v3_large

# สร้างโมเดลเปล่า
model = mobilenet_v3_large(pretrained=False)
# ปรับ classifier layer ให้ตรงกับจำนวนคลาส (10 คลาสสำหรับโรคมะเขือเทศ)
model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=10)

# โหลด state_dict
try:
    # ลองโหลดแบบ state_dict
    model.load_state_dict(torch.load('mobilenetv3_large_100_checkpoint_fold4.pt', map_location=device))
except Exception as e:
    # ถ้าไม่สามารถโหลดเป็น state_dict ได้ ให้ลองโหลดแบบเต็มโมเดล
    try:
        checkpoint = torch.load('mobilenetv3_large_100_checkpoint_fold4.pt', map_location=device)
        if hasattr(checkpoint, 'state_dict'):
            model.load_state_dict(checkpoint.state_dict())
        else:
            # สมมติว่าตัวแปร checkpoint คือโมเดลทั้งหมด
            model = checkpoint
    except Exception as load_error:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {load_error}")
        st.stop()

# ตั้งค่าโมเดลให้อยู่ในโหมดประเมินผล
model.eval()


# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['Bacterial_spot', 'Early_blight', 'healthy', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_Mites', 'Target_Spot', 'mosaic_virus', 'Yellow_Leaf_Curl_Virus']

    if st.button('Prediction'):
        #Prediction class
        probli = pred_class(model,image,class_name)
        
        st.write("## Prediction Result")
        # Get the index of the maximum value in probli[0]
        max_index = np.argmax(probli[0])

        # Iterate over the class_name and probli lists
        for i in range(len(class_name)):
            # Set the color to blue if it's the maximum value, otherwise use the default color
            color = "blue" if i == max_index else None
            st.write(f"## <span style='color:{color}'>{class_name[i]} : {probli[0][i]*100:.2f}%</span>", unsafe_allow_html=True)