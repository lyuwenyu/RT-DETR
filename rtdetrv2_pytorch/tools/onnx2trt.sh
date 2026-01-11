#!/bin/bash

# A script to convert an ONNX model to a TensorRT engine using trtexec.
# This script automatically sets the output engine path based on the input ONNX file.

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if an input file is provided.
if [ -z "$1" ]; then
    echo "Error: No ONNX file provided."
    echo "Usage: $0 /path/to/your/model.onnx"
    exit 1
fi

ONNX_FILE=$1
# Replace the .onnx extension with .trt for the output file.
ENGINE_FILE="${ONNX_FILE%.onnx}.trt"

echo "==> Converting ONNX to TensorRT Engine <=="
echo "  - Input ONNX:  $ONNX_FILE"
echo "  - Output TRT:  $ENGINE_FILE"
echo "  - Precision:   FP16"
echo "=========================================="

# Run the trtexec command.
# --fp16 enables 16-bit floating-point precision for faster inference.
# --verbose provides detailed output during the conversion process.
trtexec --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16 \
        --verbose

echo "=========================================="
echo "✅ Successfully created TensorRT engine: $ENGINE_FILE"