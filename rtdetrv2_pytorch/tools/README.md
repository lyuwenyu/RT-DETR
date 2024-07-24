

Train/test script examples
- `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 tools/train.py -c path/to/config &> train.log 2>&1 &`
- `-r path/to/checkpoint`
- `--amp`
- `--test-only` 


Export script examples
- `python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check`


Gpu do not release memory
- `ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9`


Save all logs
- Appending `&> train.log 2>&1 &` or `&> train.log 2>&1`


Tensorboard
- `--summary-dir=/path/to/summary/dir` or `-u summary_dir=/path/to/summary/dir`
- `tensorboard --host=ip --port=8989 --logdir=/path/to/summary/`