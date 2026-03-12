import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

import time
import os
from supervision import Color
import shutil
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--classes', type=str, default="runnable area", help='検出したいクラス名（例: "runnable area"）')
args = parser.parse_args()

CLASSES = [args.classes]

# GroundingDINO config and checkpoint
# GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swinb_cogcoor.pth"

# Segment-Anything checkpoint
# SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
# SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = "./sam_hq_vit_h.pth"
SAM_ENCODER_VERSION = "vit_l"
SAM_CHECKPOINT_PATH = "./sam_vit_l_0b3195.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
# sam = sam_hq_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# Predict classes and hyper-param for GroundingDINO
# SOURCE_IMAGE_PATH = "./input/image_2.png"
IMAGE_DIR = "./input_robomech"

 # Directory to save results    
output_dir = f'robomech_groundedsam/{args.classes}'
# Clear output_dir if it exists, then recreate it
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove all contents of the directory
os.makedirs(output_dir, exist_ok=True)

# CLASSES = ["runnable area"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# 画像ファイル一覧を取得（jpg, png対応）
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

times = []

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    image = cv2.imread(image_path)

    # # inference
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    
    start_time = time.time()
       
    with torch.no_grad(): 
        # start.record()
        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        
        # # annotate image with detections
        # box_annotator = sv.BoxAnnotator()
        # # labels = [
        # #     f"{CLASSES[class_id]} {confidence:0.2f}" 
        # #     for _, _, confidence, class_id, _, _ 
        # #     in detections]

        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}"
        #     for class_id, confidence
        #     in zip(detections.class_id, detections.confidence)
        # ]

        # # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        # # save the annotated grounding dino image
        # cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)


        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        # end.record()
    
    end_time = time.time() - start_time
    print(f"end_time: {end_time:.6f} sec.")
    times.append(end_time)
        
    # torch.cuda.synchronize()
    # elapsed_time = start.elapsed_time(end)
    # times.append(elapsed_time / 1000)
    
    # print(elapsed_time / 1000, 'sec.')


    # annotate image with detections
    # box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color=Color(128, 64, 128), opacity=0.8)

    # labels = [
    #     f"{CLASSES[class_id]} {confidence:0.2f}" 
    #     for _, _, confidence, class_id, _, _ 
    #     in detections]


    # labels = [
    #     f"{CLASSES[class_id]} {confidence:0.2f}"
    #     for class_id, confidence
    #     in zip(detections.class_id, detections.confidence)
    # ]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    
    # 保存ファイル名を決定
    output_path = os.path.join(output_dir, f"grounded_sam_{os.path.splitext(image_file)[0]}.jpg")
    cv2.imwrite(output_path, annotated_image)

    print(f"{image_file} done.")
if len(times) > 3:
    print(f"平均処理時間: {np.mean(times[5:]):.6f} sec.")
else:
    print(f"平均処理時間: {np.mean(times):.6f} sec.")


# save the annotated grounded-sam image
# cv2.imwrite("grounded_sam_annotated_image.jpg", annotated_image)
