import tensorflow as tf
import json
from pathlib import Path
import numpy as np

def load_and_preprocess(record, img_shape=(224, 224)):
    """Load and preprocess images and annotations in parallel"""

    # Image
    img_path = record["item"]        
    img_data = tf.io.read_file(img_path)
    img = tf.io.decode_png(img_data, channels=3)
    img = tf.image.resize(img, img_shape)
    img = (img / 255.0).numpy()
    
    # Load annotation
    annotation_path = img_path.replace('.png', '.json')
    with open(annotation_path, 'r') as f:
        ann = json.load(f)
    
    bboxes = np.array(ann['objects']['bbox'], dtype=np.float32)
    categories = np.array(ann['objects']['category'], dtype=np.int32)
    
    return {"image": img, "bboxes": bboxes, "categories": categories}

def prepare_tf_batch(batch, max_objects=30):
    """Prepare batch for TensorFlow with padding to match model expectations"""
    batch_size = len(batch["image"])
    
    # Pad bboxes and categories
    padded_bboxes = np.zeros((batch_size, max_objects, 4), dtype=np.float32)
    padded_categories = np.full((batch_size, max_objects), -1, dtype=np.int32)
    
    for i in range(batch_size):
        num_objects = min(len(batch["bboxes"][i]), max_objects)
        padded_bboxes[i, :num_objects] = batch["bboxes"][i][:num_objects]
        padded_categories[i, :num_objects] = batch["categories"][i][:num_objects]
    
    return {
        "images": np.stack(batch["image"]),
        "bboxes": padded_bboxes,
        "classes": padded_categories
    }