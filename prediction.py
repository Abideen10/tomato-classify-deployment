## Making Pridcition return class & prob
from typing import List, Tuple
import torch
import torchvision.transforms as T

from PIL import Image
def pred_class(model: torch.nn.Module,
                        image,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        ):
    
    # 2. Open image
    img = image

    # 3. Create transformation for image (if one doesn't exist)
    image_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    

    ### Predict on image ### 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()  # ตั้งค่าโมเดลให้อยู่ในโหมดประเมินผล
    
    try:
        # ลองใช้ half precision ถ้าสามารถทำได้
        if device != "cpu" and torch.cuda.is_available():  # half precision มักใช้ได้กับ GPU เท่านั้น
            model.half()
            use_half = True
        else:
            use_half = False
            
        with torch.inference_mode():
          # 6. Transform and add an extra dimension to image
          transformed_image = image_transform(img).unsqueeze(dim=0)
          
          # ปรับ precision ตามที่โมเดลต้องการ
          if use_half:
              transformed_image = transformed_image.half()
          else:
              transformed_image = transformed_image.float()

          # 7. ส่งภาพไปยังอุปกรณ์ที่ใช้
          transformed_image = transformed_image.to(device)
          
          # ทำนาย
          target_image_pred = model(transformed_image)

    except RuntimeError as e:
        # ถ้าเกิดปัญหาเกี่ยวกับ half precision ให้ลองใช้ full precision
        print(f"Error using half precision: {e}. Trying with full precision...")
        model.float()  # กลับไปใช้ full precision
        
        with torch.inference_mode():
          transformed_image = image_transform(img).unsqueeze(dim=0).float().to(device)
          target_image_pred = model(transformed_image)

    # 8. Convert logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # คืนค่าผลลัพธ์การทำนาย
    prob = target_image_pred_probs.cpu().numpy()

    return prob