
"""
Utility functions for processing BitNet model weights
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from safetensors import safe_open
import torch

logger = logging.getLogger(__name__)

class WeightProcessor:
    """Handles various weight processing operations"""
    
    @staticmethod
    def detect_weight_format(weight: np.ndarray) -> str:
        """Detect the format of weight tensor"""
        unique_values = np.unique(weight)
        
        if len(unique_values) <= 3 and all(v in [-1, 0, 1] for v in unique_values):
            return "ternary"
        elif weight.dtype in [np.int8, np.uint8]:
            return "int8"
        elif weight.dtype == np.float16:
            return "fp16"
        elif weight.dtype == np.float32:
            return "fp32"
        else:
            return "unknown"
    
    @staticmethod
    def dequantize_ternary(weight: np.ndarray, scale: float = 1.0, 
                          zero_point: float = 0.0) -> np.ndarray:
        """
        Dequantize ternary weights to FP32
        
        Args:
            weight: Ternary weight tensor with values {-1, 0, 1}
            scale: Scaling factor for dequantization
            zero_point: Zero point for quantization
            
        Returns:
            Dequantized FP32 weight tensor
        """
        # Ensure input is numpy array
        if torch.is_tensor(weight):
            weight = weight.detach().cpu().numpy()
        
        # Convert to float32 and apply scaling
        dequantized = weight.astype(np.float32)
        if scale != 1.0:
            dequantized = dequantized * scale
        if zero_point != 0.0:
            dequantized = dequantized + zero_point
            
        logger.debug(f"Dequantized ternary weight: {weight.shape} -> {dequantized.shape}")
        return dequantized
    
    @staticmethod
    def extract_scale_from_name(weight_name: str, weights: Dict[str, np.ndarray]) -> float:
        """Extract scaling factor from weight name or associated scale tensor"""
        # Look for associated scale tensor
        scale_names = [
            weight_name.replace('.weight', '.scale'),
            weight_name.replace('.weight', '_scale'),
            weight_name + '_scale'
        ]
        
        for scale_name in scale_names:
            if scale_name in weights:
                scale_tensor = weights[scale_name]
                if scale_tensor.size == 1:
                    return float(scale_tensor.item())
                else:
                    # Per-channel scaling - return mean for simplification
                    return float(np.mean(scale_tensor))
        
        return 1.0  # Default scale
    
    @staticmethod
    def validate_weight_shapes(weights: Dict[str, np.ndarray], 
                             expected_shapes: Dict[str, Tuple[int, ...]]) -> bool:
        """Validate that weight shapes match expected dimensions"""
        for name, expected_shape in expected_shapes.items():
            if name in weights:
                actual_shape = weights[name].shape
                if actual_shape != expected_shape:
                    logger.warning(f"Shape mismatch for {name}: "
                                 f"expected {expected_shape}, got {actual_shape}")
                    return False
        return True
    
    @staticmethod
    def normalize_weight_names(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize weight tensor names to standard format"""
        normalized = {}
        
        name_mappings = {
            # Common variations in naming
            'tok_embeddings.weight': 'embed_tokens.weight',
            'token_emb.weight': 'embed_tokens.weight',
            'embed.weight': 'embed_tokens.weight',
            'output.weight': 'lm_head.weight',
            'classifier.weight': 'lm_head.weight',
            # Layer normalization variations
            'ln_1.weight': 'input_layernorm.weight',
            'ln_2.weight': 'post_attention_layernorm.weight',
            'norm.weight': 'input_layernorm.weight',
            # Attention projection variations
            'attn.c_attn.weight': 'self_attn.qkv_proj.weight',
            'attn.c_proj.weight': 'self_attn.o_proj.weight',
        }
        
        for original_name, weight in weights.items():
            # Apply direct mappings
            normalized_name = name_mappings.get(original_name, original_name)
            
            # Handle layer-specific naming patterns
            for pattern, replacement in name_mappings.items():
                if pattern in normalized_name:
                    normalized_name = normalized_name.replace(pattern, replacement)
            
            normalized[normalized_name] = weight
        
        logger.info(f"Normalized {len(weights)} weight tensors")
        return normalized

class SafetensorsLoader:
    """Enhanced safetensors loader with error handling and validation"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
    def load(self) -> Dict[str, np.ndarray]:
        """Load weights with comprehensive error handling"""
        try:
            weights = {}
            with safe_open(self.model_path, framework="numpy") as f:
                metadata = f.metadata()
                if metadata:
                    logger.info(f"Model metadata: {metadata}")
                
                total_params = 0
                for key in f.keys():
                    weight = f.get_tensor(key)
                    weights[key] = weight
                    total_params += weight.size
                    
                    logger.debug(f"Loaded: {key}, shape: {weight.shape}, "
                               f"dtype: {weight.dtype}, format: {WeightProcessor.detect_weight_format(weight)}")
                
                logger.info(f"Loaded {len(weights)} tensors, {total_params:,} total parameters")
                
            return weights
            
        except Exception as e:
            logger.error(f"Failed to load safetensors file: {e}")
            raise
    
    def validate_file(self) -> bool:
        """Validate safetensors file integrity"""
        try:
            with safe_open(self.model_path, framework="numpy") as f:
                keys = list(f.keys())
                if not keys:
                    logger.error("Safetensors file contains no tensors")
                    return False
                
                # Try to load first tensor as validation
                first_key = keys[0]
                first_tensor = f.get_tensor(first_key)
                if first_tensor is None:
                    logger.error("Failed to load first tensor")
                    return False
                
                logger.info(f"Safetensors file validation passed: {len(keys)} tensors")
                return True
                
        except Exception as e:
            logger.error(f"Safetensors validation failed: {e}")
            return False

def process_bitnet_weights(raw_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Main function to process BitNet weights for ONNX conversion
    
    Args:
        raw_weights: Raw weights loaded from safetensors
        
    Returns:
        Processed weights ready for ONNX conversion
    """
    logger.info("Processing BitNet weights for ONNX conversion")
    
    # Normalize weight names
    weights = WeightProcessor.normalize_weight_names(raw_weights)
    
    # Process each weight based on its type and purpose
    processed_weights = {}
    
    for name, weight in weights.items():
        weight_format = WeightProcessor.detect_weight_format(weight)
        
        if weight_format == "ternary":
            # Extract scale if available
            scale = WeightProcessor.extract_scale_from_name(name, weights)
            processed_weight = WeightProcessor.dequantize_ternary(weight, scale)
            logger.debug(f"Dequantized ternary weight: {name} (scale: {scale})")
        else:
            # Convert to FP32
            processed_weight = weight.astype(np.float32)
            logger.debug(f"Converted to FP32: {name} ({weight_format})")
        
        processed_weights[name] = processed_weight
    
    logger.info(f"Processed {len(processed_weights)} weight tensors")
    return processed_weights
