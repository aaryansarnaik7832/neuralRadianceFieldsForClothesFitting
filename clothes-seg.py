import numpy as np
import torch
import cv2
import argparse
import os
from pathlib import Path
from typing import List, Tuple
from segment_anything import sam_model_registry, SamPredictor

def get_arguments():
    parser = argparse.ArgumentParser(description="Segment objects of interest using SAM")
    parser.add_argument("--input_dir", type=str, default="./input", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output images")
    parser.add_argument("--resize_images", action="store_true", help="Resize input images to specified dimensions")
    parser.add_argument("--image_size", type=str, default='480_720', help="Target image size (width_height)")
    parser.add_argument("--keypoint_pos", type=str, default='center', help="Positive keypoint location")
    parser.add_argument("--use_negative_keypoint", action="store_true", help="Use negative keypoint for segmentation")
    parser.add_argument("--keypoint_neg", type=str, default='10_10', help="Negative keypoint location (width_height)")
    return parser.parse_args()

def load_sam_model(checkpoint_path: str, model_type: str, device: str):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)

def preprocess_images(image_paths: List[str], target_size: Tuple[int, int] = None) -> List[np.ndarray]:
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            image = cv2.resize(image, target_size)
        images.append(image)
    return images

def get_keypoints(image_shape: Tuple[int, int, int], pos_keypoint: str, neg_keypoint: str = None) -> Tuple[np.ndarray, np.ndarray]:
    height, width, _ = image_shape
    if pos_keypoint == 'center':
        pos_point = np.array([[width // 2, height // 2]])
    else:
        raise ValueError(f"Unsupported positive keypoint location: {pos_keypoint}")
    
    pos_label = np.array([1])
    
    if neg_keypoint is not None:
        neg_width, neg_height = map(int, neg_keypoint.split("_"))
        neg_point = np.array([[neg_width, neg_height]])
        neg_label = np.array([0])
        pos_point = np.concatenate([pos_point, neg_point])
        pos_label = np.concatenate([pos_label, neg_label])
    
    return pos_point, pos_label

def segment_images(predictor: SamPredictor, images: List[np.ndarray], keypoints: np.ndarray, labels: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    masks, isolated_objects = [], []
    for idx, image in enumerate(images, start=1):
        predictor.set_image(image)
        masks_pred, scores_pred, _ = predictor.predict(
            point_coords=keypoints,
            point_labels=labels,
            multimask_output=True,
        )
        best_mask = masks_pred[scores_pred.argmax()]
        best_score = scores_pred.max()
        
        if best_score < 0.98:
            print(f"Skipping image {idx} due to low confidence score.")
            continue
        
        background_mask = 1 - best_mask[:, :, np.newaxis]
        isolated_object = (best_mask[:, :, np.newaxis] * image) + (background_mask * 255).astype(np.uint8)
        
        masks.append(best_mask)
        isolated_objects.append(isolated_object)
        
        print(f"Processed image {idx}/{len(images)}...")
    
    return masks, isolated_objects

def save_results(output_dir: str, isolated_objects: List[np.ndarray]):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, isolated_object in enumerate(isolated_objects, start=1):
        isolated_object = cv2.cvtColor(isolated_object, cv2.COLOR_RGB2BGR)
        output_file = output_path / f"{idx:04d}.png"
        cv2.imwrite(str(output_file), isolated_object)
        print(f"Saved image {idx}/{len(isolated_objects)}...")

def main():
    args = get_arguments()
    
    checkpoint_path = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = load_sam_model(checkpoint_path, model_type, device)
    
    image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.png'))]
    image_paths.sort()
    
    if args.resize_images:
        target_size = tuple(map(int, args.image_size.split("_")))
        images = preprocess_images(image_paths[:3], target_size)
    else:
        images = preprocess_images(image_paths[:3])
    
    keypoints, labels = get_keypoints(images[0].shape, args.keypoint_pos, args.keypoint_neg if args.use_negative_keypoint else None)
    
    masks, isolated_objects = segment_images(predictor, images, keypoints, labels)
    
    save_results(args.output_dir, isolated_objects)
    
    print("=" * 60)
    print("Segmentation completed successfully!")

if __name__ == '__main__':
    main()