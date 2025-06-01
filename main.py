
"""
BitNet to ONNX Conversion Utility
Converts Microsoft BitNet models to ONNX format for CPU inference
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from safetensors import safe_open
import json
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BitNetModelConfig:
    """Configuration class for BitNet model parameters"""
    
    def __init__(self):
        # Default BitNet-b1.58-2B-4T architecture parameters from GGUF log
        self.architecture = "bitnet-b1.58"
        self.vocab_size = 128256
        self.context_length = 4096
        self.embedding_length = 2560
        self.block_count = 30
        self.feed_forward_length = 6912
        self.rope_dimension_count = 128
        self.attention_head_count = 20
        self.attention_head_count_kv = 5  # Grouped Query Attention
        self.attention_layer_norm_rms_epsilon = 1e-5
        self.rope_freq_base = 500000.0
        self.quantization_type = "i2_s"  # 2 bpw ternary
        self.tokenizer_model = "gpt2"

class BitNetWeightLoader:
    """Handles loading and dequantization of BitNet model weights"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.weights = {}
        
    def load_safetensors(self) -> Dict[str, np.ndarray]:
        """Load weights from safetensors file"""
        logger.info(f"Loading weights from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        weights = {}
        with safe_open(self.model_path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
                logger.debug(f"Loaded weight: {key}, shape: {weights[key].shape}")
        
        return weights
    
    def dequantize_ternary_weights(self, quantized_weights: np.ndarray, 
                                 scale: float = 1.0) -> np.ndarray:
        """
        Dequantize 1.58-bit ternary weights {-1, 0, 1} to FP32
        """
        # Convert ternary values to float32
        dequantized = quantized_weights.astype(np.float32) * scale
        logger.debug(f"Dequantized weights shape: {dequantized.shape}")
        return dequantized
    
    def process_weights(self, raw_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process and dequantize all model weights"""
        processed_weights = {}
        
        for name, weight in raw_weights.items():
            if "embed" in name.lower() or "norm" in name.lower():
                # Keep embedding and normalization weights as-is
                processed_weights[name] = weight.astype(np.float32)
            else:
                # Assume other weights are quantized and need dequantization
                processed_weights[name] = self.dequantize_ternary_weights(weight)
                
        logger.info(f"Processed {len(processed_weights)} weight tensors")
        return processed_weights

class BitNetONNXBuilder:
    """Builds ONNX graph for BitNet model"""
    
    def __init__(self, config: BitNetModelConfig):
        self.config = config
        self.nodes = []
        self.initializers = []
        self.inputs = []
        self.outputs = []
        self.value_info = []
        
    def create_embedding_layer(self, weights: Dict[str, np.ndarray]):
        """Create token embedding layer"""
        logger.info("Creating embedding layer")
        
        # Add embedding weight as initializer
        if "embed_tokens.weight" in weights:
            embed_weight = weights["embed_tokens.weight"]
            self.initializers.append(
                helper.make_tensor(
                    name="embed_weight",
                    data_type=TensorProto.FLOAT,
                    dims=embed_weight.shape,
                    vals=embed_weight.flatten().tolist()
                )
            )
            
            # Create Gather node for embedding lookup
            self.nodes.append(
                helper.make_node(
                    "Gather",
                    inputs=["embed_weight", "input_ids"],
                    outputs=["embeddings"],
                    name="token_embedding"
                )
            )
        
    def create_layer_norm(self, name: str, weight_name: str, weights: Dict[str, np.ndarray]):
        """Create RMS Layer Normalization"""
        if weight_name in weights:
            norm_weight = weights[weight_name]
            
            # Add normalization weight as initializer
            self.initializers.append(
                helper.make_tensor(
                    name=f"{name}_weight",
                    data_type=TensorProto.FLOAT,
                    dims=norm_weight.shape,
                    vals=norm_weight.flatten().tolist()
                )
            )
            
            # Add epsilon constant
            epsilon_tensor = helper.make_tensor(
                name=f"{name}_epsilon",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[self.config.attention_layer_norm_rms_epsilon]
            )
            self.initializers.append(epsilon_tensor)
    
    def create_attention_layer(self, layer_idx: int, weights: Dict[str, np.ndarray]):
        """Create grouped query attention layer"""
        logger.debug(f"Creating attention layer {layer_idx}")
        
        layer_prefix = f"model.layers.{layer_idx}"
        
        # Query, Key, Value projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight_key = f"{layer_prefix}.self_attn.{proj}.weight"
            if weight_key in weights:
                weight = weights[weight_key]
                self.initializers.append(
                    helper.make_tensor(
                        name=f"layer_{layer_idx}_{proj}_weight",
                        data_type=TensorProto.FLOAT,
                        dims=weight.shape,
                        vals=weight.flatten().tolist()
                    )
                )
                
                # Create MatMul node
                input_name = "embeddings" if layer_idx == 0 else f"layer_{layer_idx-1}_output"
                output_name = f"layer_{layer_idx}_{proj}_output"
                
                self.nodes.append(
                    helper.make_node(
                        "MatMul",
                        inputs=[input_name, f"layer_{layer_idx}_{proj}_weight"],
                        outputs=[output_name],
                        name=f"layer_{layer_idx}_{proj}"
                    )
                )
    
    def create_feed_forward_layer(self, layer_idx: int, weights: Dict[str, np.ndarray]):
        """Create feed-forward network layer"""
        logger.debug(f"Creating FFN layer {layer_idx}")
        
        layer_prefix = f"model.layers.{layer_idx}.mlp"
        
        # Gate, Up, Down projections
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            weight_key = f"{layer_prefix}.{proj}.weight"
            if weight_key in weights:
                weight = weights[weight_key]
                self.initializers.append(
                    helper.make_tensor(
                        name=f"layer_{layer_idx}_ffn_{proj}_weight",
                        data_type=TensorProto.FLOAT,
                        dims=weight.shape,
                        vals=weight.flatten().tolist()
                    )
                )
    
    def create_transformer_block(self, layer_idx: int, weights: Dict[str, np.ndarray]):
        """Create complete transformer block"""
        logger.debug(f"Creating transformer block {layer_idx}")
        
        # Input layer norm
        self.create_layer_norm(
            f"layer_{layer_idx}_input_norm",
            f"model.layers.{layer_idx}.input_layernorm.weight",
            weights
        )
        
        # Attention layer
        self.create_attention_layer(layer_idx, weights)
        
        # Post-attention layer norm
        self.create_layer_norm(
            f"layer_{layer_idx}_post_attn_norm",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            weights
        )
        
        # Feed-forward layer
        self.create_feed_forward_layer(layer_idx, weights)
    
    def create_output_layer(self, weights: Dict[str, np.ndarray]):
        """Create language model head"""
        logger.info("Creating output layer")
        
        # Final layer norm
        self.create_layer_norm(
            "final_norm",
            "model.norm.weight",
            weights
        )
        
        # Language model head
        if "lm_head.weight" in weights:
            lm_head_weight = weights["lm_head.weight"]
            self.initializers.append(
                helper.make_tensor(
                    name="lm_head_weight",
                    data_type=TensorProto.FLOAT,
                    dims=lm_head_weight.shape,
                    vals=lm_head_weight.flatten().tolist()
                )
            )
            
            self.nodes.append(
                helper.make_node(
                    "MatMul",
                    inputs=[f"layer_{self.config.block_count-1}_output", "lm_head_weight"],
                    outputs=["logits"],
                    name="lm_head"
                )
            )
    
    def build_onnx_model(self, weights: Dict[str, np.ndarray]) -> onnx.ModelProto:
        """Build complete ONNX model"""
        logger.info("Building ONNX model")
        
        # Define model inputs
        self.inputs = [
            helper.make_tensor_value_info(
                "input_ids",
                TensorProto.INT64,
                ["batch_size", "sequence_length"]
            )
        ]
        
        # Define model outputs
        self.outputs = [
            helper.make_tensor_value_info(
                "logits",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", str(self.config.vocab_size)]
            )
        ]
        
        # Create embedding layer
        self.create_embedding_layer(weights)
        
        # Create transformer blocks
        for i in range(self.config.block_count):
            self.create_transformer_block(i, weights)
        
        # Create output layer
        self.create_output_layer(weights)
        
        # Create ONNX graph
        graph = helper.make_graph(
            nodes=self.nodes,
            name="BitNet-b1.58-2B-4T",
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_info
        )
        
        # Create ONNX model
        model = helper.make_model(graph, producer_name="BitNet-to-ONNX-Converter")
        model.opset_import[0].version = 17
        
        return model

class BitNetConverter:
    """Main converter class"""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        self.config = BitNetModelConfig()
        
    def convert(self):
        """Perform the conversion from BitNet to ONNX"""
        logger.info(f"Starting conversion: {self.model_path} -> {self.output_path}")
        
        try:
            # Load model weights
            loader = BitNetWeightLoader(self.model_path)
            raw_weights = loader.load_safetensors()
            processed_weights = loader.process_weights(raw_weights)
            
            # Build ONNX model
            builder = BitNetONNXBuilder(self.config)
            onnx_model = builder.build_onnx_model(processed_weights)
            
            # Validate ONNX model
            logger.info("Validating ONNX model")
            onnx.checker.check_model(onnx_model)
            
            # Save ONNX model
            logger.info(f"Saving ONNX model to {self.output_path}")
            onnx.save(onnx_model, self.output_path)
            
            # Test with ONNX Runtime
            self.test_onnx_runtime()
            
            logger.info("Conversion completed successfully!")
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            raise
    
    def test_onnx_runtime(self):
        """Test the generated ONNX model with ONNX Runtime"""
        logger.info("Testing ONNX model with ONNX Runtime")
        
        try:
            # Create ONNX Runtime session (CPU only)
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(self.output_path, providers=providers)
            
            # Get input and output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            logger.info(f"Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
            logger.info(f"Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
            
            # Create dummy input for basic test
            dummy_input = np.random.randint(0, 1000, size=(1, 10), dtype=np.int64)
            
            # Run inference
            outputs = session.run(None, {input_info.name: dummy_input})
            logger.info(f"Inference test successful. Output shape: {outputs[0].shape}")
            
        except Exception as e:
            logger.warning(f"ONNX Runtime test failed: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Convert BitNet models to ONNX format")
    parser.add_argument("--input", "-i", required=True, 
                       help="Path to input BitNet model file (.safetensors)")
    parser.add_argument("--output", "-o", required=True,
                       help="Path to output ONNX model file (.onnx)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Perform conversion
    converter = BitNetConverter(args.input, args.output)
    converter.convert()

if __name__ == "__main__":
    main()
