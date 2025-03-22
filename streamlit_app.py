import streamlit as st 
import torch
import torch.nn as nn
from PIL import Image
from prediction import pred_class
import numpy as np
from torchvision.models import mobilenet_v3_large

# Set title 
st.title('Tomato Disease Classification')

#Set Header 
st.header('Please up load picture')

# แสดงข้อมูล PyTorch เวอร์ชัน
st.write(f"PyTorch version: {torch.__version__}")

#Load Model 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {device}")

# สร้างโมเดลเปล่า
model = mobilenet_v3_large(pretrained=False)
# ปรับ classifier layer ให้เหมาะกับจำนวนคลาส (10 คลาสสำหรับโรคมะเขือเทศ)
model.classifier[3] = nn.Linear(in_features=1280, out_features=10)

# พยายามโหลดโมเดล
try:
    # ลองโหลดเป็น state_dict ก่อน
    try:
        state_dict = torch.load('model_state_dict.pt', map_location=device)
        model.load_state_dict(state_dict)
        st.success("โหลดโมเดลจาก state_dict สำเร็จ")
    except FileNotFoundError:
        # ถ้าไม่มีไฟล์ state_dict ลองโหลดจากไฟล์โมเดลดั้งเดิม
        try:
            checkpoint = torch.load('mobilenetv3_large_100_checkpoint_fold4.pt', map_location=device)
            
            # ตรวจสอบว่าเป็น state_dict หรือโมเดลทั้งตัว
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                st.success("โหลดโมเดลจาก checkpoint['state_dict'] สำเร็จ")
            elif isinstance(checkpoint, dict):
                # ถ้าเป็น dict แต่ไม่มี 'state_dict'
                model.load_state_dict(checkpoint)
                st.success("โหลดโมเดลจาก dict สำเร็จ")
            else:
                # สมมติว่าเป็นโมเดลทั้งตัว
                model = checkpoint
                st.success("โหลดโมเดลทั้งตัวสำเร็จ (ไม่แนะนำ)")
                
            # บันทึกเป็น state_dict สำหรับใช้ในอนาคต
            torch.save(model.state_dict(), 'model_state_dict.pt')
            st.info("บันทึก state_dict สำหรับใช้ในอนาคตแล้ว")
        except Exception as model_error:
            st.error(f"ไม่สามารถโหลดโมเดลจากไฟล์ดั้งเดิมได้: {model_error}")
            st.stop()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
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
        with st.spinner('กำลังวิเคราะห์ภาพ...'):
            probli = pred_class(model, image, class_name)
        
        st.write("## Prediction Result")
        # Get the index of the maximum value in probli[0]
        max_index = np.argmax(probli[0])

        # Iterate over the class_name and probli lists
        for i in range(len(class_name)):
            # Set the color to blue if it's the maximum value, otherwise use the default color
            color = "blue" if i == max_index else None
            st.write(f"## <span style='color:{color}'>{class_name[i]} : {probli[0][i]*100:.2f}%</span>", unsafe_allow_html=True)