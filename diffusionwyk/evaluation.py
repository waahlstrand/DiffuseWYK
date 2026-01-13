# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Optional
import torch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Instances

__all__ = ["KnownBoxFilteredCOCOEvaluator"]

logger = logging.getLogger(__name__)


class KnownBoxFilteredCOCOEvaluator(COCOEvaluator):
    """
    COCO evaluator that excludes known boxes from evaluation metrics.

    This is useful when evaluating models that leverage known (ground-truth) boxes
    during inference. By filtering them out, we measure only the model's ability
    to detect previously unknown objects.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the evaluator. Arguments are the same as COCOEvaluator.
        """
        super().__init__(*args, **kwargs)
        self._filtered_count = 0
        self._total_count = 0

    def process(self, inputs: List[Dict], outputs: List[Dict]):
        """
        Process predictions, filtering out known boxes before evaluation.

        Args:
            inputs: List of input dicts, may contain 'known_mask' field
            outputs: List of output dicts containing 'instances' predictions
        """
        filtered_outputs = []

        for input_dict, output_dict in zip(inputs, outputs):
            output = output_dict.copy()

            # Check if this sample has known boxes to filter
            if "known_mask" in input_dict and "instances" in output_dict:
                pred_instances = output_dict["instances"]
                known_mask = input_dict["known_mask"]

                # Ensure tensors are on same device and dtype
                if isinstance(known_mask, torch.Tensor):
                    known_mask = known_mask.to(pred_instances.pred_boxes.device)
                else:
                    known_mask = torch.as_tensor(
                        known_mask,
                        device=pred_instances.pred_boxes.device,
                        dtype=torch.bool,
                    )

                # Keep only non-known predictions
                # Truncate known_mask to match number of predictions
                num_preds = len(pred_instances)
                known_mask = known_mask[:num_preds]

                if known_mask.any():
                    keep = ~known_mask
                    filtered_instances = pred_instances[keep]

                    # Log filtering statistics
                    num_removed = (~keep).sum().item()
                    self._filtered_count += num_removed
                    self._total_count += num_preds

                    if num_removed > 0:
                        logger.debug(
                            f"Filtered {num_removed} known boxes from {num_preds} predictions"
                        )

                    output["instances"] = filtered_instances
                else:
                    self._total_count += num_preds

            filtered_outputs.append(output)

        # Call parent process with filtered outputs
        super().process(inputs, filtered_outputs)

    def evaluate(self):
        """
        Evaluate and return results.
        """
        results = super().evaluate()

        # Log filtering statistics
        if self._total_count > 0:
            logger.info(
                f"Known box filtering statistics: "
                f"Filtered {self._filtered_count} boxes out of {self._total_count} total predictions "
                f"({100.0 * self._filtered_count / self._total_count:.1f}%)"
            )

        return results
