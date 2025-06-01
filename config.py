
"""
Configuration module for BitNet model parameters and conversion settings
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os

@dataclass
class BitNetArchConfig:
    """BitNet architecture configuration"""
    architecture: str = "bitnet-b1.58"
    vocab_size: int = 128256
    context_length: int = 4096
    embedding_length: int = 2560
    block_count: int = 30
    feed_forward_length: int = 6912
    rope_dimension_count: int = 128
    attention_head_count: int = 20
    attention_head_count_kv: int = 5
    attention_layer_norm_rms_epsilon: float = 1e-5
    rope_freq_base: float = 500000.0
    quantization_type: str = "i2_s"
    tokenizer_model: str = "gpt2"

@dataclass
class ConversionConfig:
    """Conversion process configuration"""
    target_precision: str = "fp32"
    onnx_opset_version: int = 17
    enable_optimization: bool = False
    validate_model: bool = True
    test_runtime: bool = True
    cpu_threads: Optional[int] = None

class ConfigManager:
    """Manages configuration loading and saving"""
    
    @staticmethod
    def load_from_file(config_path: str) -> tuple[BitNetArchConfig, ConversionConfig]:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            return BitNetArchConfig(), ConversionConfig()
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        arch_config = BitNetArchConfig(**config_data.get('architecture', {}))
        conv_config = ConversionConfig(**config_data.get('conversion', {}))
        
        return arch_config, conv_config
    
    @staticmethod
    def save_to_file(arch_config: BitNetArchConfig, conv_config: ConversionConfig, 
                    config_path: str):
        """Save configuration to JSON file"""
        config_data = {
            'architecture': arch_config.__dict__,
            'conversion': conv_config.__dict__
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @staticmethod
    def create_default_config(config_path: str):
        """Create default configuration file"""
        arch_config = BitNetArchConfig()
        conv_config = ConversionConfig()
        ConfigManager.save_to_file(arch_config, conv_config, config_path)

# Default configurations for different model variants
MODEL_CONFIGS = {
    "bitnet-b1.58-2b-4t": BitNetArchConfig(
        architecture="bitnet-b1.58",
        vocab_size=128256,
        context_length=4096,
        embedding_length=2560,
        block_count=30,
        feed_forward_length=6912,
        rope_dimension_count=128,
        attention_head_count=20,
        attention_head_count_kv=5,
        attention_layer_norm_rms_epsilon=1e-5,
        rope_freq_base=500000.0,
        quantization_type="i2_s",
        tokenizer_model="gpt2"
    )
}

def get_model_config(model_name: str) -> BitNetArchConfig:
    """Get predefined model configuration by name"""
    return MODEL_CONFIGS.get(model_name.lower(), BitNetArchConfig())
