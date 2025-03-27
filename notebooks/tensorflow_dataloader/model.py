from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Permute, Reshape

def masked_mse(y_true, y_pred):
    mask = tf.reduce_sum(tf.abs(y_true), axis=-1) > 0
    return tf.reduce_mean(tf.boolean_mask(tf.square(y_true - y_pred), mask))

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    mask = y_true >= 0
    return tf.keras.losses.sparse_categorical_crossentropy(
        tf.boolean_mask(y_true, mask),
        tf.boolean_mask(y_pred, mask)
    )


def build_object_detection_model(num_classes=5, max_objects=30, input_shape = (224, 224, 3)):
    inputs = Input(shape=input_shape, name='images')
    x = Permute((2, 3, 1))(inputs)

    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        alpha=1.0
    )
    backbone.trainable = True
    
    # Detection head
    x = backbone.output
    
    # Bounding box branch (4 coordinates per object)
    bbox_branch = Conv2D(256, (3,3), padding='same')(backbone.output)
    bbox_out = Conv2D(max_objects*4, (1,1))(bbox_branch)
    bbox_out = GlobalAveragePooling2D()(bbox_out)
    bbox_out = Reshape((max_objects, 4), name='bboxes')(bbox_out)

    # Class prediction branch
    cls_branch = Conv2D(256, (3,3), padding='same')(backbone.output) 
    cls_out = Conv2D(max_objects*num_classes, (1,1))(cls_branch)
    cls_out = GlobalAveragePooling2D()(cls_out)
    cls_out = Reshape((max_objects, num_classes), name='classes')(cls_out)

    # Create model
    model = Model(inputs=inputs, outputs=[bbox_out, cls_out])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'bboxes': masked_mse,
            'classes': masked_sparse_categorical_crossentropy
        },
        metrics={
            'bboxes': ['mae', masked_mse],
            'classes': ['accuracy', masked_sparse_categorical_crossentropy]
        }
    )

    return model

def process_predictions(pred, confidence_threshold=0.5, num_classes=5):
    """
    Processes raw model outputs into valid categorical predictions
    with confidence filtering.
    
    Args:
        bbox_preds: Array of shape (batch, 30, 4) - bounding boxes
        class_preds: Array of shape (batch, 30, 5) - class logits
        confidence_threshold: Minimum probability to consider a detection valid
        
    Returns:
        filtered_results: List of lists containing valid (class, bbox, confidence)
    """
    bbox_preds = pred[0]
    class_preds = pred[1]
    
    # Convert logits to probabilities
    class_probs = tf.nn.softmax(class_preds, axis=-1).numpy()
    
    # Get predicted class and confidence
    pred_classes = np.argmax(class_probs, axis=-1)  # Shape: (batch, 30)
    confidences = np.max(class_probs, axis=-1)      # Shape: (batch, 30)
    
    # Initialize container for filtered results
    filtered_results = []
    
    # Process each sample in batch
    for batch_idx in range(bbox_preds.shape[0]):
        batch_results = []
        
        # Process each object prediction
        for obj_idx in range(bbox_preds.shape[1]):
            cls = pred_classes[batch_idx, obj_idx]
            conf = confidences[batch_idx, obj_idx]
            bbox = bbox_preds[batch_idx, obj_idx]
            
            # Filter based on confidence and valid class
            if conf >= confidence_threshold and cls < num_classes:
                batch_results.append({
                    'class': int(cls),
                    'confidence': float(conf),
                    'bbox': bbox.tolist()
                })
                
        filtered_results.append(batch_results)
    
    return filtered_results
