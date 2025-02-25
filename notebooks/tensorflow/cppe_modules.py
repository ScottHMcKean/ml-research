import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

IMG_HEIGHT = 224
IMG_WIDTH = 224
MAX_OBJECTS = 20
NUM_CLASSES = 5

def create_model():
    """
    Create a custom CNN model for object detection
    """
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Feature extraction layers
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    # For each potential object: 4 bbox coords + num_classes + 1 objectness score
    output_size = MAX_OBJECTS * (4 + NUM_CLASSES + 1)
    outputs = tf.keras.layers.Dense(output_size)(x)
    outputs = tf.keras.layers.Reshape((MAX_OBJECTS, 4 + NUM_CLASSES + 1))(outputs)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
  

def parse_annotation(annotation):
    """
    Parse COCO format annotations [x_min, y_min, width, height]
    and normalize to [0, 1] scale
    """
    boxes = np.zeros((MAX_OBJECTS, 4))
    classes = np.zeros(MAX_OBJECTS)
    mask = np.zeros(MAX_OBJECTS)
    
    for idx, _ in enumerate(annotation['objects']['id']):
        if idx >= MAX_OBJECTS:
            break
            
        # Get COCO format bbox
        x_min, y_min, width, height = annotation['objects']['bbox'][idx]
        
        # Convert to normalized coordinates [x_min, y_min, x_max, y_max]
        x_min_norm = x_min / annotation['width']
        y_min_norm = y_min / annotation['height']
        x_max_norm = (x_min + width) / annotation['width']
        y_max_norm = (y_min + height) / annotation['height']
        
        # Clip values to [0, 1]
        x_min_norm = np.clip(x_min_norm, 0, 1)
        y_min_norm = np.clip(y_min_norm, 0, 1)
        x_max_norm = np.clip(x_max_norm, 0, 1)
        y_max_norm = np.clip(y_max_norm, 0, 1)
        
        boxes[idx] = [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
        classes[idx] = annotation['objects']['category'][idx]
        mask[idx] = 1
    
    return boxes, classes, mask
  

class ObjectDetectionDataset:
    def __init__(self, image_dir, batch_size=16, max_files=None):
        self.max_files = max_files
        self.image_dir = image_dir
        self.batch_size = batch_size
        
    def load_data(self):
        images = []
        all_boxes = []
        all_classes = []
        all_masks = []
        
        if self.max_files is None:
            self.max_files = len(os.listdir(self.image_dir))

        for img_file in os.listdir(self.image_dir)[:self.max_files]:
            if img_file.endswith('.png'):
                
                # Load image and get original dimensions
                img_path = os.path.join(self.image_dir, img_file)
                img = Image.open(img_path)
                
                # Convert to RGB if not already
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = np.array(img) / 255.0
                
                # Verify shape is correct
                assert img_array.shape == (IMG_HEIGHT, IMG_WIDTH, 3), f"Incorrect shape for {img_file}: {img_array.shape}"
                
                # Load annotation
                json_file = img_file.replace('.png', '.json')
                json_path = os.path.join(self.image_dir, json_file)
                with open(json_path, 'r') as f:
                    annotation = json.load(f)
                
                boxes, classes, mask = parse_annotation(annotation)
                
                images.append(img_array)
                all_boxes.append(boxes)
                all_classes.append(classes)
                all_masks.append(mask)
        
        # Convert to numpy arrays
        images = np.array(images)
        all_boxes = np.array(all_boxes)
        all_classes = np.array(all_classes)
        all_masks = np.array(all_masks)
        
        print(f"Loaded {len(images)} images with shape {images.shape}")
        
        return images, all_boxes, all_classes, all_masks
    
    def create_tf_dataset(self, images, boxes, classes, masks):
        """Create a tf.data.Dataset with batching and shuffling"""
        # Convert classes to one-hot encoding
        classes_one_hot = tf.keras.utils.to_categorical(classes, num_classes=NUM_CLASSES)
        
        # Combine targets
        y = np.concatenate([
            boxes,
            classes_one_hot,
            np.expand_dims(masks, -1)
        ], axis=-1)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, y))
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
      
def compute_iou(boxes1, boxes2):
    """Compute IOU between two sets of boxes"""
    # Calculate intersection coordinates
    x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
    
    # Calculate area of intersection
    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    
    # Calculate area of both boxes
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Calculate IoU
    union = area1 + area2 - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())
    
    return iou

def custom_loss(y_true, y_pred):
    """Custom loss function using IoU for bounding boxes"""
    # Unpack the true values
    true_boxes = y_true[:, :, :4]
    true_classes = y_true[:, :, 4:-1]
    true_mask = y_true[:, :, -1]
    
    # Unpack predictions
    pred_boxes = y_pred[:, :, :4]
    pred_classes = y_pred[:, :, 4:-1]
    pred_objectness = y_pred[:, :, -1]
    
    # Box loss using IoU
    iou = compute_iou(true_boxes, pred_boxes)
    box_loss = 1 - iou
    box_loss = tf.reduce_mean(box_loss * true_mask)
    
    # Class loss
    class_loss = tf.keras.losses.categorical_crossentropy(
        true_classes, pred_classes, from_logits=True)
    class_loss = tf.reduce_mean(class_loss * true_mask)
    
    # Objectness loss
    obj_loss = tf.keras.losses.binary_crossentropy(
        true_mask, pred_objectness, from_logits=True)
    obj_loss = tf.reduce_mean(obj_loss)
    
    return box_loss + class_loss + obj_loss