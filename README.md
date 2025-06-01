
# BitNet to ONNX Conversion Utility

A Python-based utility for converting Microsoft BitNet language models to ONNX format for CPU inference.

## Overview

This tool converts BitNet-b1.58-2B-4T models from their native format (safetensors) to ONNX format, enabling CPU-based inference using standard ONNX runtimes. The converter handles the dequantization of 1.58-bit ternary weights to FP32 format.

## Prerequisites

- Python 3.9+
- CPU with AVX/AVX2 support (no GPU required)
- OpenSUSE Tumbleweed (or compatible Linux distribution)

## Installation

The required dependencies are automatically managed through the project configuration:

- onnx >= 1.14.0
- onnxruntime >= 1.15.0
- numpy >= 1.21.0
- safetensors >= 0.3.0
- torch >= 2.0.0
- transformers >= 4.46.3
- protobuf >= 4.21.0
- sentencepiece >= 0.2.0

## Usage

### Basic Conversion

```bash
python main.py --input /path/to/bitnet/model.safetensors --output /path/to/output/model.onnx
```

### With Verbose Logging

```bash
python main.py --input /path/to/bitnet/model.safetensors --output /path/to/output/model.onnx --verbose
```

### Command Line Arguments

- `--input, -i`: Path to input BitNet model file (.safetensors)
- `--output, -o`: Path to output ONNX model file (.onnx)
- `--verbose, -v`: Enable verbose logging for debugging

## Model Architecture

The converter handles the following BitNet-b1.58-2B-4T architecture parameters:

- **Vocabulary Size**: 128,256 tokens
- **Context Length**: 4,096 tokens
- **Embedding Dimensions**: 2,560
- **Transformer Blocks**: 30 layers
- **Feed-Forward Dimensions**: 6,912
- **Attention Heads**: 20 (with 5 KV heads for Grouped Query Attention)
- **RoPE Dimensions**: 128
- **Quantization**: 1.58-bit ternary weights (dequantized to FP32)

## Input Requirements

The converter expects:

1. **Model Weights**: A `.safetensors` file containing the BitNet model weights
2. **Weight Format**: Weights should follow the standard transformer naming convention:
   - `embed_tokens.weight`: Token embeddings
   - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`: Attention projections
   - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`: Feed-forward projections
   - `model.layers.{i}.{input,post_attention}_layernorm.weight`: Layer normalizations
   - `model.norm.weight`: Final layer normalization
   - `lm_head.weight`: Language model head

## Output

The converter produces:

1. **ONNX Model**: A `.onnx` file with FP32 weights suitable for CPU inference
2. **Validation**: Automatic model validation using ONNX checker
3. **Runtime Test**: Basic compatibility test with ONNX Runtime

## Limitations

- Weights are dequantized to FP32 (no native 1.58-bit support in ONNX)
- CPU-only inference (no GPU optimizations)
- Basic structural conversion (limited numerical equivalency testing)
- Tokenizer logic not included in ONNX model

## Architecture Details

The converter implements:

- **Token Embeddings**: Standard embedding lookup
- **Transformer Blocks**: Multi-head attention with RMSNorm
- **Grouped Query Attention**: Efficient attention mechanism
- **Feed-Forward Networks**: Standard MLP with gate/up/down projections
- **RoPE**: Rotary positional embeddings
- **Language Model Head**: Final projection to vocabulary

## Troubleshooting

### Common Issues

1. **Missing weights**: Ensure the input model contains all required weight tensors
2. **Memory errors**: Large models may require sufficient RAM for conversion
3. **ONNX validation errors**: Check that all weight dimensions are compatible

### Logging

Use the `--verbose` flag to enable detailed logging for debugging conversion issues.

## Technical Notes

- The converter uses ONNX opset version 17
- All computations are performed in FP32 precision
- The model is optimized for CPU execution with standard ONNX operators
- Ternary weight dequantization preserves the original {-1, 0, 1} values scaled to FP32

## License

This utility is provided as-is for research and development purposes.
