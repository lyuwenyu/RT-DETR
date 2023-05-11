# 论文测速使用的部分代码和工具

## RT-DETR TensorRT部署 [in progress]
以[rtdetr-r50](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)为例
1. 转换到onnx  
2. 转tensorrt engine
3. Python脚本测速或者部署


## 测试YOLO系列的速度 [in progress]
以[yolov8](https://github.com/ultralytics/ultralytics)为例
1. 转onnx  
执行`yolov8_onnx.py`中的`export_onnx`函数，新增代码主要涉及输出格式的转换

2. onnx插入nms算子  
使用`utils.py`中的`yolo_insert_nms`函数，导出onnx模型后使用[Netron](https://netron.app/)查看结构. <img width="924" alt="image" src="https://github.com/lyuwenyu/RT-DETR/assets/17582080/cb466483-d3a3-4f23-a68d-7ab8825059c8">

3. 转tensorrt engine
可以使用`trtexec.md`中的的脚本转换，或者使用`utils.py`中的Python代码转换
```bash
# trtexec -h
trtexec --onnx=./yolov8l_w_nms.onnx --saveEngine=yolov8l_w_nms.engine --buildOnly --fp16
```

4. 使用trtexec测速
可以使用`trtexec.md`中的的脚本转换，去掉`--buildOnly`参数。同时也可以使用nsys可视化profile timeline
<img width="1090" alt="image" src="https://github.com/lyuwenyu/RT-DETR/assets/17582080/507d8bde-9e7c-4ae5-b571-976c540ef2c6">


5. Python脚本测速或者部署
在Coco val数据集上测模型的平均速度使用`trtinfer.py`中的代码推理