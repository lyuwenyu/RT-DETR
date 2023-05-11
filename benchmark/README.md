# 论文测速使用的部分代码和工具


## 测试YOLO系列的速度 [in progress]
以[yolov8](https://github.com/ultralytics/ultralytics)为例

<details open>
<summary>1. 转onnx </summary>  

执行`yolov8_onnx.py`中的`export_onnx`函数，新增代码主要涉及输出格式的转换
</details>


<details>
<summary>2. 插入nms </summary>

使用`utils.py`中的`yolo_insert_nms`函数，导出onnx模型后使用[Netron](https://netron.app/)查看结构. <img width="924" alt="image" src="https://github.com/lyuwenyu/RT-DETR/assets/17582080/cb466483-d3a3-4f23-a68d-7ab8825059c8">
</details>


<details>
<summary>3. 转tensorrt </summary>

可以使用`trtexec.md`中的的脚本转换，或者使用`utils.py`中的Python代码转换
```bash
# trtexec -h
trtexec --onnx=./yolov8l_w_nms.onnx --saveEngine=yolov8l_w_nms.engine --buildOnly --fp16
```
</details>


<details>
<summary>4. trtexec测速 </summary>

可以使用`trtexec.md`中的的脚本转换，去掉`--buildOnly`参数

</details>



<details>
<summary>5. profile分析（可选） </summary>

在4的基础之上加以下命令
```bash
nsys profile --force-overwrite=true  -t 'nvtx,cuda,osrt,cudnn' -c cudaProfilerApi -o yolov8l_w_nms 
```
可以使用nsys可视化分析
<img width="1090" alt="image" src="https://github.com/lyuwenyu/RT-DETR/assets/17582080/507d8bde-9e7c-4ae5-b571-976c540ef2c6">

</details>


<details>
<summary>6. Python测速或者部署   </summary>

在Coco val数据集上测模型的平均速度使用`trtinfer.py`中的代码推理

</details>
