#
import logging
from typing import Dict, List, Optional
import torch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Instances
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
import os
import pickle

__all__ = ["KnownBoxFilteredCOCOEvaluator", "KnownBoxSeparatedCOCOEvaluator"]

logger = logging.getLogger(__name__)


class KnownBoxFilteredCOCOEvaluator(COCOEvaluator):
    """
    COCO evaluator that excludes known boxes from evaluation metrics.

    This is useful when evaluating models that leverage known (ground-truth) boxes
    during inference. By filtering them out, we measure only the model's ability
    to detect previously unknown objects.
    """

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(
                    prediction["proposals"].objectness_logits.numpy()
                )

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(
                os.path.join(self._output_dir, "box_proposals.pkl"), "wb"
            ) as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(
                    predictions, self._coco_api, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res


class KnownBoxSeparatedCOCOEvaluator(COCOEvaluator):
    """
    COCO evaluator that separately evaluates known vs. unknown boxes.

    This evaluator keeps all boxes in the ground truth (known + unknown) and
    reports separate metrics for:
    - Known boxes: boxes provided as input to the model
    - Unknown boxes: boxes the model must discover
    - Overall: all boxes combined

    This allows measuring both how well the model refines known boxes and
    how well it discovers unknown objects.
    """

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions, reporting metrics separately
        for known and unknown boxes.
        """
        if self._output_dir:
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(
                    prediction["proposals"].objectness_logits.numpy()
                )

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(
                os.path.join(self._output_dir, "box_proposals.pkl"), "wb"
            ) as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals with known/unknown separation...")
        res_all = {}
        res_known = {}
        res_unknown = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}

        for limit in [100, 1000]:
            for area, suffix in areas.items():
                # Evaluate all boxes
                stats_all = _evaluate_box_proposals_separated(
                    predictions,
                    self._coco_api,
                    area=area,
                    limit=limit,
                    known_filter=None,
                )
                key_all = "AR{}@{:d}".format(suffix, limit)
                res_all[key_all] = float(stats_all["ar"].item() * 100)

                # Evaluate only known boxes
                stats_known = _evaluate_box_proposals_separated(
                    predictions,
                    self._coco_api,
                    area=area,
                    limit=limit,
                    known_filter=True,
                )
                key_known = "AR{}@{:d}".format(suffix, limit)
                res_known[key_known] = float(stats_known["ar"].item() * 100)

                # Evaluate only unknown boxes
                stats_unknown = _evaluate_box_proposals_separated(
                    predictions,
                    self._coco_api,
                    area=area,
                    limit=limit,
                    known_filter=False,
                )
                key_unknown = "AR{}@{:d}".format(suffix, limit)
                res_unknown[key_unknown] = float(stats_unknown["ar"].item() * 100)

        # Log results in organized sections
        self._logger.info(
            "=== Overall Metrics (All Boxes) ===\n" + create_small_table(res_all)
        )
        self._logger.info(
            "=== Known Boxes Metrics ===\n" + create_small_table(res_known)
        )
        self._logger.info(
            "=== Unknown Boxes Metrics (Discovery) ===\n"
            + create_small_table(res_unknown)
        )

        # Store combined results
        combined_res = {}
        for k, v in res_all.items():
            combined_res[f"all_{k}"] = v
        for k, v in res_known.items():
            combined_res[f"known_{k}"] = v
        for k, v in res_unknown.items():
            combined_res[f"unknown_{k}"] = v

        self._results["box_proposals"] = combined_res

        print("Box proposal evaluation results (overall, known, unknown):")
        print(create_small_table(combined_res))


# inspired from Detectron:
# https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
def _evaluate_box_proposals(
    dataset_predictions, coco_api, thresholds=None, area="all", limit=None
):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0**2, 1e5**2],  # all
        [0**2, 32**2],  # small
        [32**2, 96**2],  # medium
        [96**2, 1e5**2],  # large
        [96**2, 128**2],  # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)

        # Remove known boxes from ground truth
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if (obj["iscrowd"] == 0 and obj["known"] == False)
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)

        # Remove known boxes from ground truth areas
        gt_areas = torch.as_tensor(
            [
                obj["area"]
                for obj in anno
                if (obj["iscrowd"] == 0 and obj["known"] == False)
            ]
        )

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0)
        if len(gt_overlaps)
        else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def _evaluate_box_proposals_separated(
    dataset_predictions,
    coco_api,
    thresholds=None,
    area="all",
    limit=None,
    known_filter=None,
):
    """
    Evaluate detection proposal recall metrics separately for known and unknown boxes.

    Args:
        known_filter: None (all boxes), True (only known), False (only unknown)
    """
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0**2, 1e5**2],  # all
        [0**2, 32**2],  # small
        [32**2, 96**2],  # medium
        [96**2, 1e5**2],  # large
        [96**2, 128**2],  # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)

        # Filter ground truth based on known_filter
        if known_filter is None:
            # Use all boxes
            filter_condition = lambda obj: obj["iscrowd"] == 0
        elif known_filter is True:
            # Use only known boxes
            filter_condition = (
                lambda obj: obj["iscrowd"] == 0 and obj.get("known", False) == True
            )
        else:  # known_filter is False
            # Use only unknown boxes
            filter_condition = (
                lambda obj: obj["iscrowd"] == 0 and obj.get("known", False) == False
            )

        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if filter_condition(obj)
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
        gt_boxes = Boxes(gt_boxes)

        gt_areas = torch.as_tensor(
            [obj["area"] for obj in anno if filter_condition(obj)]
        )

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            box_ind = argmax_overlaps[gt_ind]
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        gt_overlaps.append(_gt_overlaps)

    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0)
        if len(gt_overlaps)
        else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        recalls[i] = (
            (gt_overlaps >= t).float().sum() / float(num_pos) if num_pos > 0 else 0.0
        )
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }
