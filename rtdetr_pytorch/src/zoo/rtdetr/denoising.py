"""by lyuwenyu
"""

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh



def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device
    
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # if label_noise_ratio > 0:
    #     input_query_class = input_query_class.flatten()
    #     pad_gt_mask = pad_gt_mask.flatten()
    #     # half of bbox prob
    #     # mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
    #     mask = torch.rand_like(input_query_class) < (label_noise_ratio * 0.5)
    #     chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
    #     # randomly put a new one here
    #     new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
    #     # input_query_class.scatter_(dim=0, index=chosen_idx, value=new_label)
    #     input_query_class[chosen_idx] = new_label
    #     input_query_class = input_query_class.reshape(bs, num_denoising)
    #     pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # class_embed = torch.concat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=device)])
    # input_query_class = torch.gather(
    #     class_embed, input_query_class.flatten(),
    #     axis=0).reshape(bs, num_denoising, -1)
    # input_query_class = class_embed(input_query_class.flatten()).reshape(bs, num_denoising, -1)
    input_query_class = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True
    
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])
    
    return input_query_class, input_query_bbox, attn_mask, dn_meta
