import argparse
import sys
import base64
import csv
import numpy as np
import mmcv
import torch
import torch.nn.functional as F
#from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.datasets import build_dataloader, DirectoryBasedDataset
from mmdet.core import wrap_fp16_model, delta2bbox
from mmdet.models import build_detector
from mmdet.ops.nms import nms_wrapper

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",  "num_boxes", "boxes", "features"]
csv.field_size_limit(sys.maxsize)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--max_features', type=int, required=False, default=100)
    parser.add_argument('--min_features', type=int, required=False, default=1)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument(
        '--tsv_out',
        help='output result file name without extension',
        type=str, required=True)
    args = parser.parse_args()
    return args


def multiclass_nms_with_ids_preserving(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):

    num_classes = multi_scores.shape[1]
    bboxes, labels, keep_ids = [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, cls_det_inds = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ), i, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        keep_ids.append(cls_det_inds)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        keep_ids = torch.cat(keep_ids)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            keep_ids = keep_ids[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        keep_ids = multi_bboxes.new_zeros((0,), dtype=torch.long)

    return bboxes, labels, keep_ids


def postprocess_features(model, rois, roi_feats, bbox_preds, cls_score, max_boxes, img_shape, normalize_boxes=True):
    if isinstance(cls_score, list):
        cls_score = sum(cls_score) / float(len(cls_score))
    scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

    if bbox_preds is not None:
        bboxes = delta2bbox(rois[:, 1:], bbox_preds, model.bbox_head.target_means,
                            model.bbox_head.target_stds, img_shape)
    else:
        bboxes = rois[:, 1:].clone()
        if img_shape is not None:
            bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
            bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)
    cfg = model.test_cfg.rcnn
    det_bboxes, det_labels, keep = multiclass_nms_with_ids_preserving(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
    det_roi_feats = roi_feats[keep]
    det_scores = scores[keep]
    if normalize_boxes:
        bboxes[:, [0, 2]] /= img_shape[1]
        bboxes[:, [1, 3]] /= img_shape[0]
    if len(det_labels) > max_boxes:
        keep_boxes = np.argsort(det_scores)[::-1][:max_boxes]
        det_bboxes = det_bboxes[keep_boxes]
        det_labels = det_labels[keep_boxes]
        det_scores = det_scores[keep_boxes]
        det_roi_feats = det_roi_feats[keep_boxes]

    # min det case
    return {
        'objects_id': base64.b64encode(det_labels),
        'objects_conf': base64.b64encode(det_scores),
        'num_boxes': len(det_labels),
        'boxes': base64.b64encode(det_bboxes),
        'features': base64.b64encode(det_roi_feats)
    }


def extract_features(model, data_loader, max_obj=100):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            rois, roi_feats, bbox_pred, cls_scores = model.extract_all_features(**data)
            img_shape = data['img_meta'][0]['img_shape']
            scale_factor = data['img_meta'][0]['scale_factor']
            height = img_shape[0] / scale_factor
            width = img_shape[1] / scale_factor
            result_dict = {
                'img_id': dataset.image_ids[i],
                'img_h': height,
                'img_w': width
            }
            result_dict.update(postprocess_features(model, rois, roi_feats, bbox_pred, cls_scores, max_obj, img_shape))
            results.append(result_dict)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    dataset = DirectoryBasedDataset(args.data_dir, cfg.data.test.pipeline)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']

 #   model = MMDataParallel(model, device_ids=[0])
    results = extract_features(model, data_loader, args.max_features)

    print('\nwriting results to {}'.format(args.tsv_out))
    with open(args.tsv_out, "r") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        writer.writerows(results)


if __name__ == '__main__':
    main()
