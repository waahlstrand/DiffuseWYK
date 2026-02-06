# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import random
from typing import List
from collections import namedtuple

from matplotlib import patches
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_normalize_xyxy
from .util.misc import nested_tensor_from_tensor_list

__all__ = ["DiffusionWYK"]

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# @META_ARCH_REGISTRY.register()
class DiffusionDetBase(nn.Module):
    """
    Implement DiffusionWYK
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionWYK.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionWYK.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionWYK.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionWYK.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.DiffusionWYK.SAMPLE_STEP
        self.objective = "pred_x0"
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.0
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionWYK.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        class_weight = cfg.MODEL.DiffusionWYK.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionWYK.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionWYK.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionWYK.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionWYK.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionWYK.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionWYK.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionWYK.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg,
            cost_class=class_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal=self.use_focal,
        )
        weight_dict = {
            "loss_ce": class_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
        }
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg,
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            use_focal=self.use_focal,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def model_predictions(
        self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False
    ):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[
            -1
        ]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.0) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(
        self,
        batched_inputs,
        backbone_feats,
        images_whwh,
        images,
        clip_denoised=True,
        do_postprocess=True,
    ):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = (
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(
                backbone_feats,
                images_whwh,
                img,
                time_cond,
                self_cond,
                clip_x_start=clip_denoised,
            )
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = (
                    outputs_class[-1][0],
                    outputs_coord[-1][0],
                )
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat(
                    (
                        img,
                        torch.randn(
                            1, self.num_proposals - num_remain, 4, device=img.device
                        ),
                    ),
                    dim=1,
                )
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(
                    outputs_class[-1], outputs_coord[-1], images.image_sizes
                )
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(
                    box_pred_per_image, scores_per_image, labels_per_image, 0.5
                )
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]

            outputs_class, outputs_coord = self.head(features, x_boxes, t, None)
            output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

            if self.deep_supervision:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor(
                [[0.5, 0.5, 1.0, 1.0]], dtype=torch.float, device=self.device
            )
            num_gt = 1

        num_repeat = (
            self.num_proposals // num_gt
        )  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [
            num_repeat + 1
        ] * (self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2.0 - 1.0) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.0

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor(
                [[0.5, 0.5, 1.0, 1.0]], dtype=torch.float, device=self.device
            )
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = (
                torch.randn(self.num_proposals - num_gt, 4, device=self.device) / 6.0
                + 0.5
            )  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (
                num_gt - self.num_proposals
            )
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2.0 - 1.0) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.0

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device
            )
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return (
            new_targets,
            torch.stack(diffused_boxes),
            torch.stack(noises),
            torch.stack(ts),
        )

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = (
                torch.arange(self.num_classes, device=self.device)
                .unsqueeze(0)
                .repeat(self.num_proposals, 1)
                .flatten(0, 1)
            )

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(
                zip(scores, box_pred, image_sizes)
            ):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(
                    self.num_proposals, sorted=False
                )
                labels_per_image = labels[topk_indices]
                box_pred_per_image = (
                    box_pred_per_image.view(-1, 1, 4)
                    .repeat(1, self.num_classes, 1)
                    .view(-1, 4)
                )
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image, 0.5
                    )
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (
                scores_per_image,
                labels_per_image,
                box_pred_per_image,
                image_size,
            ) in enumerate(zip(scores, labels, box_pred, image_sizes)):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image, 0.5
                    )
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(
                torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
            )
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh


@META_ARCH_REGISTRY.register()
class DiffusionWYK(DiffusionDetBase):
    """
    Implement DiffusionWYK, diffusion of what you know.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_known_train = (
            cfg.MODEL.DiffusionWYK.NUM_KNOWN_TRAIN
        )  # Number of known labels
        self.num_known_test = (
            cfg.MODEL.DiffusionWYK.NUM_KNOWN_TEST
        )  # Number of known labels at test time
        self.known_noise_level = cfg.MODEL.DiffusionWYK.KNOWN_NOISE_LEVEL

        self.num_test_proposals = cfg.MODEL.DiffusionWYK.NUM_TEST_PROPOSALS
        self.ddim_sampling_eta = cfg.MODEL.DiffusionWYK.ETA

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)

            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]

            outputs_class, outputs_coord = self.head(features, x_boxes, t, None)
            output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

            if self.deep_supervision:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device
            )
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return (
            new_targets,
            torch.stack(diffused_boxes),
            torch.stack(noises),
            torch.stack(ts),
        )

    def prepare_diffusion_concat(self, gt_boxes):
        """
        For each image, randomly select a number of known boxes (up to num_known_train),
        keep them fixed during diffusion, and diffuse the rest boxes.

        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]

        # decide a random number of known boxes for this image (0..max_known)
        max_known = min(self.num_known_train, num_gt)
        if max_known > 0:
            # sample uniformly in [0, max_known]
            num_known = int(
                torch.randint(0, max_known + 1, (1,), device=self.device).item()
            )
        else:
            num_known = 0

        # handle empty ground-truth case first
        if not num_gt:
            gt_boxes = torch.as_tensor(
                [[0.5, 0.5, 1.0, 1.0]], dtype=torch.float, device=self.device
            )
            num_gt = 1

        # Build x_start so that its length == self.num_proposals.
        # Two cases:
        # 1) num_gt <= num_proposals: x_start = [gt_boxes, placeholders]
        #    known indices should be sampled from the first num_gt entries.
        # 2) num_gt > num_proposals: randomly select which gt_boxes to keep,
        #    then sample known indices among kept positions (indices in range(num_proposals)).
        if num_gt <= self.num_proposals:
            # pad with placeholders
            if num_gt < self.num_proposals:
                box_placeholder = (
                    torch.randn(self.num_proposals - num_gt, 4, device=self.device)
                    / 6.0
                    + 0.5
                )
                box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
                x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
            else:
                x_start = gt_boxes

            # choose random known indices among the gt portion (0 .. num_gt-1)
            if num_known > 0:
                max_known = min(num_known, num_gt)
                num_known_to_select = torch.randint(
                    0, max_known + 1, (1,), device=self.device
                ).item()
                perm = torch.randperm(num_gt, device=self.device)
                known_idx = perm[:num_known_to_select]
            else:
                known_idx = torch.empty((0,), dtype=torch.long, device=self.device)

        else:
            # select a random subset of gt boxes to match num_proposals
            perm = torch.randperm(num_gt, device=self.device)
            keep = perm[: self.num_proposals]
            x_start = gt_boxes[keep]

            # choose known indices among the kept positions (0..num_proposals-1)
            if num_known > 0:
                k = min(num_known, x_start.shape[0])
                perm2 = torch.randperm(x_start.shape[0], device=self.device)
                known_idx = perm2[:k]
            else:
                known_idx = torch.empty((0,), dtype=torch.long, device=self.device)

        # scale to model coordinate space
        x_start = (x_start * 2.0 - 1.0) * self.scale

        # noise sample: only diffuse unknown boxes (zero noise for known boxes)
        if num_known > 0 and known_idx.numel() > 0:
            noise[known_idx] = 0.0
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.0

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    @torch.no_grad()
    def ddim_sample(
        self,
        batched_inputs,
        backbone_feats,
        images_whwh,
        images,
        clip_denoised=True,
        do_postprocess=True,
    ):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = (
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # Get known boxes from batched_inputs via the 'instances' and 'known_mask' field
        if self.num_known_test > 0:

            x_known_batch = []
            for i, bi in enumerate(batched_inputs):

                assert (
                    "instances" in bi
                ), "Input batch must contain 'instances' with 'known_mask' field for known boxes."
                instances = bi["instances"]

                assert hasattr(
                    instances, "known_mask"
                ), "Instances must have 'known_mask' field for known boxes."
                known_mask = instances.known_mask
                known_boxes = instances.gt_boxes.tensor[known_mask].to(self.device)
                x_known_batch.append(known_boxes)
        else:
            # Add empty known boxes if no known boxes are specified
            x_known_batch = [
                torch.empty((0, 4), device=self.device) for _ in range(batch)
            ]

        x_known_batch = torch.stack(x_known_batch)
        # Randomly initialize the starting boxes as standard normal
        x = torch.randn(shape, device=self.device)

        # Add a distribution of noisy known boxes into img at each step
        if self.num_known_test > 0 and self.num_test_proposals > 0:
            for i in range(batch):
                x_known = x_known_batch[i]
                num_known_boxes = x_known.shape[0]
                h = images_whwh[i, 1]
                w = images_whwh[i, 0]
                if num_known_boxes > 0:
                    x_known = x_known.repeat(
                        self.num_test_proposals, 1
                    )  # (num_known_boxes * N, 4)
                    num_test_proposals = self.num_test_proposals * num_known_boxes

                    # Forward diffusion to add noise
                    x_known = box_normalize_xyxy(x_known, w=w, h=h)
                    x_known = box_xyxy_to_cxcywh(x_known)
                    x_known = (x_known * 2.0 - 1.0) * self.scale

                    if self.known_noise_level > 0:
                        # Add initial noise to known boxes
                        noise_init = (
                            torch.randn(x_known.shape[0], 4, device=self.device)
                            * self.known_noise_level
                        )
                        x_known += noise_init

                    noise = torch.randn(x_known.shape[0], 4, device=self.device)
                    x_known_noisy = self.q_sample(
                        x_start=x_known,
                        t=torch.full(
                            (x_known.shape[0],),
                            times[0],
                            device=self.device,
                        ),
                        noise=noise,
                    )

                    x_known_noisy = torch.clamp(
                        x_known_noisy, min=-1 * self.scale, max=self.scale
                    )
                    x_known_noisy = ((x_known_noisy / self.scale) + 1) / 2.0

                    # Insert noisy known boxes into x
                    assert (
                        num_test_proposals <= self.num_proposals
                    ), "num_test_proposals must be <= num_proposals"
                    x[i, :num_test_proposals, :] = x_known_noisy

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:

            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(
                backbone_feats,
                images_whwh,
                x,
                time_cond,
                self_cond,
                clip_x_start=clip_denoised,
            )
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = (
                    outputs_class[-1][0],
                    outputs_coord[-1][0],
                )
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = (
                    value
                    > threshold | torch.arange(x.shape[1], device=x.device)
                    < self.num_test_proposals
                )  # keep all boxes above threshold, plus the known boxes (first num_test_proposals entries)

                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                x = x[:, keep_idx, :]
            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                x = torch.cat(
                    (
                        x,
                        torch.randn(
                            1, self.num_proposals - num_remain, 4, device=x.device
                        ),
                    ),
                    dim=1,
                )
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(
                    outputs_class[-1], outputs_coord[-1], images.image_sizes
                )
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(
                    box_pred_per_image, scores_per_image, labels_per_image, 0.5
                )

                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results
