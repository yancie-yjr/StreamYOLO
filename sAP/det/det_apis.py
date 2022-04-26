'''
APIs for interfacing detectors (mainly mmdetection)
Note that only a subset of methods are supported right now
'''

from functools import partial
import warnings
from os.path import basename
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import pycocotools.mask as maskUtils

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes, bbox2roi, bbox_mapping, merge_aug_masks
from mmdet.models import build_detector, \
    SingleStageDetector, TwoStageDetector, StandardRoIHead, \
    CascadeRoIHead


# Below are modified functions of (an old version of)
# mmdetection's preprocessing pipeline

class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose and move to GPU
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True, 
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = False # ignores input, assuming already in RGB
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True, device='cuda:0'):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device).unsqueeze(0)

        return img, img_shape, pad_shape, scale_factor

class ImageTransformGPU(object):
    """Preprocess an image ON A GPU.
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.std_inv = 1/self.std
        # self.to_rgb = to_rgb, assuming already in RGB
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True, device='cuda:0'):
        h, w = img.shape[:2]
        if keep_ratio:
            if isinstance(scale, (float, int)):
                if scale <= 0:
                    raise ValueError(
                         'Invalid scale {}, must be positive.'.format(scale))
                scale_factor = scale
            elif isinstance(scale, tuple):
                max_long_edge = max(scale)
                max_short_edge = min(scale)
                scale_factor = min(max_long_edge / max(h, w),
                                max_short_edge / min(h, w))
            else:
                raise TypeError(
                    'Scale must be a number or tuple of int, but got {}'.format(
                        type(scale)))
            
            new_size = (round(h*scale_factor), round(w*scale_factor))
        else:
            new_size = scale
            w_scale = new_size[1] / w
            h_scale = new_size[0] / h
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = (*new_size, 3)

        img = torch.from_numpy(img).to(device).float()
        # to BxCxHxW
        img = img.permute(2, 0, 1).unsqueeze_(0)

        if new_size[0] != img.shape[1] or new_size[1] != img.shape[2]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore the align_corner warnings
                img = F.interpolate(img, new_size, mode='bilinear')
        if flip:
            img = torch.flip(img, 3)
        for c in range(3):
            img[:, c, :, :].sub_(self.mean[c]).mul_(self.std_inv[c])

        if self.size_divisor is not None:
            pad_h = int(np.ceil(new_size[0] / self.size_divisor)) * self.size_divisor - new_size[0]
            pad_w = int(np.ceil(new_size[1] / self.size_divisor)) * self.size_divisor - new_size[1]
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            pad_shape = (img.shape[2], img.shape[3], 3)
        else:
            pad_shape = img_shape
        return img, img_shape, pad_shape, scale_factor


# Below are modified functions of (an old version of)
# mmdetection's detectors. Note that instead of creating 
# standalone detectors, we apply modular patches to mmdetection

def _single_stage_test(self, img, img_metas, rescale=False, numpy_res=True, decode_mask=True):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    bbox_list = self.bbox_head.get_bboxes(
        *outs, img_metas, rescale=rescale)
    det_bboxes, det_labels = bbox_list[0]
    if numpy_res:
        det_bboxes = det_bboxes.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
    return det_bboxes, det_labels
SingleStageDetector.simple_test = _single_stage_test

def _two_stage_test(self, img, img_metas, proposals=None, rescale=False, numpy_res=True, decode_mask=True):
    """simple_test without bbox2result"""
    assert self.with_bbox, "Bbox head must be implemented."

    x = self.extract_feat(img)
    
    if proposals is None:
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
    else:
        proposal_list = proposals

    return self.roi_head.simple_test(
        x, proposal_list, img_metas,
        rescale=rescale, numpy_res=numpy_res,
    )
TwoStageDetector.simple_test = _two_stage_test 

def _roi_test(self, x, proposal_list, img_metas,
    proposals=None, rescale=False, numpy_res=True):

    det_bboxes, det_labels = self.simple_test_bboxes(
        x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
    # remove the batch dimension
    det_bboxes = det_bboxes[0]
    det_labels = det_labels[0]

    if self.with_mask:
        segm_results = self.simple_test_mask(
            x, img_metas, det_bboxes, det_labels, rescale=rescale,
        )

    if numpy_res:
        det_bboxes = det_bboxes.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        if self.with_mask:
            # segm_results = mmcv.concat_list(segm_results)
            segm_results = np.asarray(segm_results)

    if self.with_mask:
        return det_bboxes, det_labels, segm_results
    else:
        return det_bboxes, det_labels
StandardRoIHead.simple_test = _roi_test

def _cascade_roi_simple_test(self, x, proposal_list, img_metas,
    proposals=None, rescale=False, numpy_res=True):
    """Test without augmentation."""
    assert self.with_bbox, 'Bbox head must be implemented.'
    num_imgs = len(proposal_list)
    img_shapes = tuple(meta['img_shape'] for meta in img_metas)
    ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
    scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

    # "ms" in variable names means multi-stage
    ms_bbox_result = {}
    ms_segm_result = {}
    ms_scores = []
    rcnn_test_cfg = self.test_cfg

    rois = bbox2roi(proposal_list)
    for i in range(self.num_stages):
        bbox_results = self._bbox_forward(i, x, rois)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(
            len(proposals) for proposals in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        if isinstance(bbox_pred, torch.Tensor):
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        else:
            bbox_pred = self.bbox_head[i].bbox_pred_split(
                bbox_pred, num_proposals_per_img)
        ms_scores.append(cls_score)

        if i < self.num_stages - 1:
            bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
            rois = torch.cat([
                self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                    bbox_pred[j],
                                                    img_metas[j])
                for j in range(num_imgs)
            ])

    # average scores of each image by stages
    cls_score = [
        sum([score[i] for score in ms_scores]) / float(len(ms_scores))
        for i in range(num_imgs)
    ]

    # apply bbox post-processing to each image individually
    det_bboxes = []
    det_labels = []
    for i in range(num_imgs):
        det_bbox, det_label = self.bbox_head[-1].get_bboxes(
            rois[i],
            cls_score[i],
            bbox_pred[i],
            img_shapes[i],
            scale_factors[i],
            rescale=rescale,
            cfg=rcnn_test_cfg)
        det_bboxes.append(det_bbox)
        det_labels.append(det_label)

    if torch.onnx.is_in_onnx_export():
        return det_bboxes, det_labels
    # bbox_results = [
    #     bbox2result(det_bboxes[i], det_labels[i],
    #                 self.bbox_head[-1].num_classes)
    #     for i in range(num_imgs)
    # ]
    # ms_bbox_result['ensemble'] = bbox_results
    ms_bbox_result['ensemble'] = [det_bboxes[0], det_labels[0]]


    if self.with_mask:
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            mask_classes = self.mask_head[-1].num_classes
            segm_results = [[[] for _ in range(mask_classes)]
                            for _ in range(num_imgs)]
        else:
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            num_mask_rois_per_img = tuple(
                _bbox.size(0) for _bbox in _bboxes)
            aug_masks = []
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                mask_pred = mask_results['mask_pred']
                # split batch mask prediction back to each image
                mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                aug_masks.append(
                    [m.sigmoid().cpu().numpy() for m in mask_pred])

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[]
                            for _ in range(self.mask_head[-1].num_classes)])
                else:
                    aug_mask = [mask[i] for mask in aug_masks]
                    merged_masks = merge_aug_masks(
                        aug_mask, [[img_metas[i]]] * self.num_stages,
                        rcnn_test_cfg)
                    segm_result = self.mask_head[-1].get_seg_masks(
                        merged_masks, _bboxes[i], det_labels[i],
                        rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
        ms_segm_result['ensemble'] = segm_results

    # if self.with_mask:
    #     results = list(
    #         zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
    # else:
    #     results = ms_bbox_result['ensemble']
    # return results

    if numpy_res:
        det_bboxes, det_labels = ms_bbox_result['ensemble']
        det_bboxes = det_bboxes.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        if self.with_mask:
            masks = ms_segm_result['ensemble'][0]
            # if deocde_mask:
            #     # masks = mmcv.concat_list(masks)
            #     masks = np.asarray(masks)

    if self.with_mask:
        return det_bboxes, det_labels, masks
    else:
        return det_bboxes, det_labels
CascadeRoIHead.simple_test = _cascade_roi_simple_test

def init_detector(opts, device='cuda:0'):
    config = mmcv.Config.fromfile(opts.config)
    new_config = 'train_pipeline' in config or 'test_pipeline' in config
    if new_config:
        # simulate old config
        if opts.in_scale is None:
            print('Warning: using new config and fixing size_divisor to 32')
            config.data.test.img_scale = config.test_pipeline[1]['img_scale']
        else:
            config.data.test.img_scale = 1
        config.data.test.size_divisor = 32
    if opts.in_scale is not None:
        if 'ssd' in basename(opts.config):
            # SSD
            if opts.in_scale <= 0.2:
                # too small leads to some issues
                l = round(1920*opts.in_scale)
                config.data.test.img_scale = (l, l)
                config.data.test.resize_keep_ratio = False
            else:
                config.data.test.img_scale = opts.in_scale
                config.data.test.resize_keep_ratio = True
        else:
            config.data.test.img_scale = opts.in_scale
            config.data.test.resize_keep_ratio = True
    if opts.no_mask:
        if 'roi_head' in config.model and 'mask_head' in config.model['roi_head']:
            config.model['roi_head']['mask_head'] = None
    config.model.pretrained = None

    model = build_detector(config.model, test_cfg=config.test_cfg)
    if opts.weights is not None:
        weights = load_checkpoint(model, opts.weights, map_location='cpu' if device == 'cpu' else None)
        if 'CLASSES' in weights['meta']:
            model.CLASSES = weights['meta']['CLASSES']
        else:
            model.CLASSES = get_classes('coco')
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True),
        device=device,
    )
    # for update in bbox_head.py
    if type(scale_factor) is int:
        scale_factor = float(scale_factor)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_metas=[img_meta])

def inference_detector(model, img, gpu_pre=True, numpy_res=True, decode_mask=True):
    # assume img has RGB channel ordering instead of BGR
    cfg = model.cfg
    if gpu_pre:
        img_transform = ImageTransformGPU(
            size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    else:
        img_transform = ImageTransform(
            size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    with torch.no_grad():
        data = _prepare_data(img, img_transform, cfg, device)
        result = model(return_loss=False, rescale=True, numpy_res=numpy_res, decode_mask=decode_mask, **data) 

    return result

if __name__ == "__main__":
    from PIL import Image
    img = np.asarray(Image.open('E:/Data/COCO/val2014/COCO_val2014_000000447342.jpg'))
    img_transform = ImageTransformGPU(
        size_divisor=32, mean=[0, 0, 0], std=[255, 255, 255])
    # img = np.zeros((416, 640, 3), np.uint8)
    img = img_transform(img,
        scale=(416, 416),
        keep_ratio=True,
        device='cuda')
    img = img
