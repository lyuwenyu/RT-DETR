
```bash
# build tensorrt engine 
trtexec --onnx=./yolov8l_w_nms.onnx --saveEngine=yolov8l_w_nms.engine --buildOnly --fp16

# using dynamic shapes
# --explicitBatch --minShapes=image:1x3x640x640 --optShapes=image:8x3x640x640  --maxShapes=image:16x3x640x640 --shapes=image:8x3x640x640

# timeline 
nsys profile --force-overwrite=true  -t 'nvtx,cuda,osrt,cudnn' -c cudaProfilerApi -o yolov8l_w_nms  trtexec --loadEngine=./yolov8l_w_nms.engine --fp16 --avgRuns=10 --loadInputs='image:input_tensor.bin'

# https://forums.developer.nvidia.com/t/about-loadinputs-in-trtexec/218880
```
