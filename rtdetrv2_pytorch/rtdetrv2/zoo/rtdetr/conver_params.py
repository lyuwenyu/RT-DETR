"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 

def main(args) -> None:
    import cvperception
    from cvperception.core import load_config, merge_config, create
    cfg = load_config(args.config)
    model: torch.nn.Module = create(cfg['model'], merge_config(cfg))

    if args.version == 1:
        state = model.state_dict()
        keys = [k for k in state.keys() if 'num_batches_tracked' not in k]

    elif args.version == 2:
        state = model.state_dict()
        ignore_keys = ['anchors', 'valid_mask', 'num_points_scale']
        keys = [k for k in state.keys() if 'num_batches_tracked' not in k]
        keys = [k for k in keys if not any([x in k for x in ignore_keys])]
    
    import paddle
    p_state = paddle.load(args.pdparams)
    pkeys = list(p_state.keys())
    
    assert len(keys) == len(pkeys), f'{len(keys)}, {len(pkeys)}'

    new_state = {}
    for i, k in enumerate(keys):    
        pp = p_state[pkeys[i]]
        pp = torch.tensor(pp.numpy())

        if 'denoising_class_embed' in k:
            new_state[k] = torch.concat([pp, torch.zeros(1, pp.shape[-1])], dim=0)
            continue

        tp = state[k]
        if len(tp.shape) == 2:
            new_state[k] = pp.T
        elif len(tp.shape) == 1:
            new_state[k] = pp
        else:
            assert tp.shape == pp.shape, f'{k}, {pp.shape}, {tp.shape}'
            new_state[k] = pp

    assert len(new_state) == len(p_state), ''

    # checkpoint = {'ema': {'module': new_state, }}
    # torch.save(checkpoint, args.output_file)

    model.load_state_dict(new_state, strict=False)

    checkpoint = {'ema': {'module': model.state_dict(), }}
    torch.save(checkpoint, args.output_file)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-p', '--pdparams', type=str, )
    parser.add_argument('-o', '--output_file', type=str, )
    parser.add_argument('-v', '--version', type=int, default=1)

    args = parser.parse_args()
    main(args)
    
    # python ./src/cvperception/zoo/rtdetr/conver_params.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -p rtdetr_r18vd_dec3_6x_coco.pdparams -o rtdetr_r18vd_dec3_6x_coco_new.pth
    # python ./src/cvperception/zoo/rtdetr/conver_params.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -p rtdetr_r18vd_5x_coco_objects365.pdparams -o rtdetr_r18vd_5x_coco_objects365_new.pth
    # python ./src/cvperception/zoo/rtdetr/conver_params.py -c configs/rtdetrv2/rtdetrv2_r50vd_120e_coco.yml -p rtdetr_r50vd_1x_objects365.pdparams -o rtdetrv2_r50vd_1x_objects365_new.pth -v 2
    
