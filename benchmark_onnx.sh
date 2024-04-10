set -x

models=()
models+=("models/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth")
models+=("models/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth")
models+=("models/rtdetr_r50vd_m_6x_coco_from_paddle.pth")
models+=("models/rtdetr_r50vd_6x_coco_from_paddle.pth")
models+=("models/rtdetr_r101vd_6x_coco_from_paddle.pth")


for model in "${models[@]}"; do
    python benchmark_onnx.py --checkpoint $model
done
