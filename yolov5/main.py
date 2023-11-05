import os
import cv2
import torch
import numpy as np
from numpy import random
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

# Load the model
weights = 'yolov5/models/trained/model.pt'  # Replace with your model weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # If CUDA isn't available, it will default to CPU
imgsz = 640  # Define your input size here
conf_thres = 0.25  # Confidence threshold
iou_thres = 0.45  # NMS IOU threshold
classes = None  # Specify classes to detect, None means all classes
agnostic_nms = False  # Apply class-agnostic NMS

# Initialize model
model = DetectMultiBackend(weights, device=device, dnn=False)
stride = model.stride
names = model.names


def extract_and_save_detections(det, original_image, names, save_dir='detections'):
    os.makedirs(save_dir, exist_ok=True)  # Create the save directory if it doesn't exist
    for idx, (*xyxy, conf, cls) in enumerate(det):
        x1, y1, x2, y2 = map(int, xyxy)
        crop_img = original_image[y1:y2, x1:x2]
        cls_name = names[int(cls)]
        save_path = os.path.join(save_dir, f"{cls_name}_{idx}.jpg")
        cv2.imwrite(save_path, crop_img)

def run_inference(image_path):
    # Ensure that the model is in evaluation mode
    model.eval()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image could not be read from the path: {image_path}")

    # Padded resize
    img = letterbox(image, imgsz, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)

    # Normalize and add batch dimension
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # add batch dimension

    # Inference
    pred = model(img, augment=False, visualize=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to image size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()

            # Extract and save detections to new images
            extract_and_save_detections(det, image, names)

            # Print results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to the image
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator = Annotator(image, line_width=3, example=label)
                annotator.box_label(xyxy, label, color=colors(int(cls), True))

    # Write results
    cv2.imwrite('inference_output1.jpg', image)  # Save annotated image

# Run inference
run_inference('yolov5/photos/image.jpg')  # Replace with the path to your image
