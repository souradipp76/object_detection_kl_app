import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import boxes as box_ops

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes, bias=True)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4, bias=True)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FastRCNNPredictorKL(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes,  bias=True)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4,  bias=True)
        self.bbox_std = nn.Linear(in_channels, num_classes * 4,  bias=True)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        bbox_stds = self.bbox_std(x)

        return scores, bbox_deltas, bbox_stds

class RoIHeadsKL(torchvision.models.detection.roi_heads.RoIHeads):

    def __init__(self, model, num_classes, cfg=None):
        super(RoIHeadsKL, self).__init__(
            model.roi_heads.box_roi_pool,
            model.roi_heads.box_head,
            #model.roi_heads.box_predictor,
            FastRCNNPredictorKL(in_channels=1024, num_classes=num_classes),
            # Faster R-CNN training
            model.roi_heads.proposal_matcher.high_threshold,
            model.roi_heads.proposal_matcher.low_threshold,
            model.roi_heads.fg_bg_sampler.batch_size_per_image,
            model.roi_heads.fg_bg_sampler.positive_fraction,
            model.roi_heads.box_coder.weights,
            # Faster R-CNN inference
            model.roi_heads.score_thresh,
            model.roi_heads.nms_thresh,
            model.roi_heads.detections_per_img
        )

        self.num_classes = num_classes

        # soft nms params
        self.softnms = cfg['softnms'] if cfg is None and 'softnms' in cfg else False
        self.softnms_sigma = 0.5

        # variance vote params
        self.var_vote = cfg['var_vote'] if cfg is not None and 'var_vote' in cfg else False
        self.var_sigma_t = 0.02

    def postprocess_detections_kl(
        self,
        class_logits,
        box_regression,
        proposals,
        image_shapes,
        box_variance
    ):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes_var = self.box_coder.decode(box_variance, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_boxes_var_list = pred_boxes_var.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, boxes_var, scores, image_shape in zip(pred_boxes_list, 
                                                         pred_boxes_var_list, 
                                                         pred_scores_list, 
                                                         image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            boxes_var = boxes_var[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # var-voting
            if self.var_vote:
                boxes, scores, labels = self.kl_nms(boxes, scores, 
                                                    self.score_thresh, 
                                                    self.nms_thresh, 
                                                    self.detections_per_img, 
                                                    boxes_var)
            else:
                # batch everything, by making every class prediction be a separate instance
                boxes = boxes.reshape(-1, 4)
                boxes_var = boxes_var.reshape(-1, 4)
                scores = scores.reshape(-1)
                labels = labels.reshape(-1)

                # remove low scoring boxes
                inds = torch.where(scores > self.score_thresh)[0]
                boxes, boxes_var, scores, labels = boxes[inds], boxes_var[inds], scores[inds], labels[inds]

                # remove empty boxes
                keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
                boxes, boxes_var, scores, labels = boxes[keep], boxes_var[keep], scores[keep], labels[keep]

                # non-maximum suppression, independently done per class
                keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
                
                # keep only topk scoring predictions
                keep = keep[: self.detections_per_img]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def kl_nms(self, bboxes, scores, score_thresh, nms_thresh, max_num, bboxes_var=None):
        bboxes = bboxes.view(-1, self.num_classes - 1, 4)
        scores = scores.view(-1, self.num_classes - 1)
        if not bboxes_var is None:
            bboxes_var = bboxes_var.view(-1, self.num_classes - 1, 4)

        def compute_iou(boxes1, boxes2):
            """
            compute IoU between boxes1 and boxes2
            """
            iou = box_ops.box_iou(boxes1, boxes2).reshape(-1)
            return iou

        def nms_class(cls_boxes, nms_iou):
            """
            Var-Voting algorithm of the original paper
            """
            assert cls_boxes.shape[1] == 5 or cls_boxes.shape[1] == 9
            keep = []
            while cls_boxes.shape[0] > 0:
                # get bbox with max score
                max_idx = torch.argmax(cls_boxes[:, 4])
                max_box = cls_boxes[max_idx].unsqueeze(0)

                # compute iou between max_box and other bboxes
                cls_boxes = torch.cat((cls_boxes[:max_idx], cls_boxes[max_idx + 1:]), 0)
                iou = compute_iou(max_box[:, :4], cls_boxes[:, :4])

                # KL var voting
                if self.var_vote and not bboxes_var is None:
                    # get overlpapped bboxes
                    iou_mask = iou > 0
                    kl_bboxes = cls_boxes[iou_mask]
                    kl_bboxes = torch.cat((kl_bboxes, max_box), dim=0)
                    kl_ious = iou[iou_mask]

                    # recover variance to sigma^2
                    kl_var = kl_bboxes[:, -4:]/8.
                    kl_var = torch.exp(kl_var)

                    # compute weighted bbox
                    p_i = torch.exp(-1*torch.pow((1 - kl_ious), 2) / self.var_sigma_t)
                    p_i = torch.cat((p_i, torch.ones(1).to(cls_boxes.device)), 0).unsqueeze(1)
                    p_i = p_i / kl_var
                    p_i = p_i / p_i.sum(dim=0)
                    max_box[0, :4] = (p_i * kl_bboxes[:, :4]).sum(dim=0)
                keep.append(max_box)

                # apply soft-NMS
                weight = torch.ones_like(iou)
                if not self.softnms:
                    weight[iou > nms_iou] = 0
                else:
                    weight = torch.exp(-1.0*(iou**2 / self.softnms_sigma))
                cls_boxes[:, 4] = cls_boxes[:, 4]*weight

                # filter bboxes with low scores
                filter_idx = (cls_boxes[:, 4] >= score_thresh).nonzero().squeeze(-1)
                cls_boxes = cls_boxes[filter_idx]
            return torch.cat(keep, 0).to(cls_boxes.device)

        # perform NMS
        output_boxes, output_scores, output_labels = [], [], []
        for i in range(self.num_classes - 1):
            filter_idx = (scores[:, i] >= score_thresh).nonzero().squeeze(-1)
            if len(filter_idx) == 0:
                continue

            filter_boxes = bboxes[filter_idx, i]
            filter_scores = scores[:, i][filter_idx].unsqueeze(1)
            if not bboxes_var is None:
                filter_boxes_var = bboxes_var[filter_idx, i]
                out_bboxes = nms_class(
                    torch.cat((filter_boxes, filter_scores, filter_boxes_var), 1), 
                    nms_thresh)
            else:
                out_bboxes = nms_class(torch.cat((filter_boxes, filter_scores), 1), nms_thresh)
            
            if out_bboxes.shape[0] > 0:
                output_boxes.append(out_bboxes[:,:4])
                output_scores.append(out_bboxes[:, 4])
                output_labels.extend([torch.ByteTensor([i+1]) for _ in range(len(out_bboxes))])

        # output results
        if len(output_boxes) == 0:
            return torch.empty(0,4).to(bboxes.device), torch.empty(0).to(scores.device), torch.empty(0).to(scores.device)
        else:
            output_boxes, output_scores, output_labels = torch.cat(output_boxes), torch.cat(output_scores), torch.cat(output_labels)

            # sort prediction
            sort_inds = torch.argsort(output_scores, descending=True)
            output_boxes, output_scores, output_labels = output_boxes[sort_inds], output_scores[sort_inds], output_labels[sort_inds]
            
            output_boxes = output_boxes[:max_num]
            output_scores = output_scores[:max_num]
            output_labels = output_labels[:max_num]
            return output_boxes, output_scores, output_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression, box_variance = self.box_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets, box_variance)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            if not box_variance is None and (self.var_vote or self.softnms):
              boxes, scores, labels = self.postprocess_detections_kl(class_logits, box_regression, proposals, image_shapes, box_variance)
            else:
              boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses

def kl_loss(bbox_pred, bbox_targets, bbox_pred_std, bbox_inside_weights=1.0, bbox_outside_weights=1.0, sigma=1.0):
    sigma_2 = sigma**2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights*box_diff #bbox_inw = in_box_diff
    bbox_l1abs = torch.abs(in_box_diff)  #abs_in_box_diff = bbox_l1abs
    smoothL1_sign = (bbox_l1abs < 1. / sigma_2).detach().float()  #1 if bbox_l1abs<1 else 0 
    bbox_inws = (torch.pow(in_box_diff, 2)*(sigma_2 / 2.)*smoothL1_sign
                + (bbox_l1abs - (0.5 / sigma_2))*(1. - smoothL1_sign)) 
    bbox_inws = bbox_inws.detach().float() 
    scale = 1
    bbox_pred_std_abs_log = bbox_pred_std*0.5*scale
    bbox_pred_std_nabs = -1.*bbox_pred_std
    bbox_pred_std_nexp = torch.exp(bbox_pred_std_nabs)
    bbox_inws_out = bbox_pred_std_nexp * bbox_inws
    bbox_pred_std_abs_logw = bbox_pred_std_abs_log*bbox_outside_weights
    bbox_pred_std_abs_logwr = torch.mean(bbox_pred_std_abs_logw, dim = 0)
    
    loss_bbox = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_pred_std_nexp)
    bbox_pred_std_abs_logw_loss = torch.sum(bbox_pred_std_abs_logwr)
    bbox_inws_out = bbox_inws_out*scale
    bbox_inws_outr = torch.mean(bbox_inws_out, dim = 0)
    bbox_pred_std_abs_mulw_loss = torch.sum(bbox_inws_outr)
    return (loss_bbox + bbox_pred_std_abs_logw_loss + bbox_pred_std_abs_mulw_loss)

def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0):    
    sigma_2 = sigma**2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights*box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    loss_box = (torch.pow(in_box_diff, 2)*(sigma_2 / 2.)*smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2))*(1. - smoothL1_sign))*bbox_outside_weights
    return loss_box.sum() / loss_box.shape[0]

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, box_variance = None):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    if not box_variance is None:
        box_variance = box_variance.reshape(N, box_variance.size(-1) // 4, 4)
        box_loss = kl_loss(box_regression[sampled_pos_inds_subset, labels_pos], 
            regression_targets[sampled_pos_inds_subset], 
            box_variance[sampled_pos_inds_subset, labels_pos], 
            bbox_inside_weights = 1.0, 
            bbox_outside_weights = 1.0,
            sigma = 3.0
        )
    else:
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

    return classification_loss, box_loss
    
def get_model(model_path = None, cfg=None):
    num_classes = 21
    in_features = 1024
    use_kl_loss = False
    if not cfg is None and 'use_kl_loss' in cfg:
        use_kl_loss = cfg['use_kl_loss']

    model = fasterrcnn_resnet50_fpn()
    if use_kl_loss:
        model.roi_heads = RoIHeadsKL(model, num_classes, cfg)
    else:
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features, num_classes=num_classes
        )

    if not model_path is None:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            print(f"Model path {model_path} does not exist.")
    return model

class_to_idx = {'_background': 0, 'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4, 'bottle':5,
        'bus':6, 'car':7, 'cat':8, 'chair':9, 'cow':10, 'diningtable':11,
        'dog':12, 'horse':13, 'motorbike':14, 'person':15, 'pottedplant':16,
        'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20
}

idx_to_class = {i:c for c, i in class_to_idx.items()}

def get_sample_prediction(model, img):
    # put the model in evaluation mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    img_t = img.resize((224, 224))
    img_t = tvt.ToTensor()(img)
    with torch.no_grad():
        prediction = model([img_t.to(device)])[0]
        
    print('predicted #boxes: ', len(prediction['labels']))
    return plot_img_bbox(img, prediction)


def plot_img_bbox(img, target, score_thres = 0.75):
    # plot the image and bboxes
    if 'scores' in target:
        classes = [idx_to_class[l.item()] for l in target['labels']]
        img = draw_boxes(target['boxes'].cpu().numpy(), classes, target['scores'].cpu().numpy(),
                         img, score_thres)

    return img


def draw_boxes(boxes, classes, scores, image, score_thres):
    FONT_SCALE = 5*1e-4  # Adjust for larger font size in all images
    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
    
    W, H = image.size
    image = np.asarray(image).astype(np.uint8)
    font_scale = max(W, H) * FONT_SCALE
    thickness = int(min(W, H) * THICKNESS_SCALE)
    for i, box in enumerate(boxes):
        box[0] = max(box[0], 0)
        box[1] = max(box[1], 0)
        box[2] = min(box[2], W)
        box[3] = min(box[3], H)
        if scores[i] > score_thres:
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0), 2
            )
            cv2.putText(image, f"{classes[i]},{scores[i]:0.2f}", (int(box[0]), int(max(20, box[1]-10))),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
    return image
