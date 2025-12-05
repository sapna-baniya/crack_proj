"""
CNN Model for Crack Detection
Custom Convolutional Neural Network architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrackDetectionCNN:
    """Custom CNN model for binary crack detection"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize CNN model
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['cnn']
        self.input_shape = tuple(self.model_config['input_shape'])
        self.num_classes = self.model_config['num_classes']
        
        self.model = None
    
    def build_model(self):
        """Build CNN architecture"""
        logger.info("Building CNN model...")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', name='conv1_1'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same', name='conv1_2'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same', name='conv2_1'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same', name='conv2_2'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same', name='conv3_1'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same', name='conv3_2'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), padding='same', name='conv4_1'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, (3, 3), padding='same', name='conv4_2'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2), name='pool4'),
            layers.Dropout(0.25),
            
            # Block 5 (Deeper features)
            layers.Conv2D(512, (3, 3), padding='same', name='last_conv_layer'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.GlobalAveragePooling2D(name='gap'),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(256, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu', name='fc2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        self.model = model
        logger.info(f"CNN model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def compile_model(self):
        """Compile the model"""
        training_config = self.config['training']
        
        optimizer = keras.optimizers.Adam(
            learning_rate=training_config['learning_rate']
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=training_config['loss_function'],
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info("Model compiled successfully")
    
    def get_model(self):
        """
        Get compiled model
        
        Returns:
            Compiled Keras model
        """
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        
        return self.model.summary()
    
    def get_layer_by_name(self, layer_name):
        """
        Get layer by name (useful for Grad-CAM)
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Keras layer
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        return self.model.get_layer(layer_name)


def main():
    """Test CNN model building"""
    # Build model
    cnn = CrackDetectionCNN()
    model = cnn.get_model()
    
    # Print summary
    print("\n" + "="*60)
    print("CNN MODEL ARCHITECTURE")
    print("="*60)
    cnn.summary()
    
    # Test forward pass
    import numpy as np
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"\nTest prediction: {output[0][0]:.4f}")


if __name__ == "__main__":
    main()