# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_diffusionwyk_config(cfg):
    """
    Add config for DiffusionWYK
    """
    cfg.MODEL.DiffusionWYK = CN()
    cfg.MODEL.DiffusionWYK.NUM_CLASSES = 80
    cfg.MODEL.DiffusionWYK.NUM_PROPOSALS = 300

    # Additions from DiffusionDet
    cfg.MODEL.DiffusionWYK.NUM_KNOWN_TRAIN = 1
    cfg.MODEL.DiffusionWYK.NUM_KNOWN_TEST = 1
    cfg.MODEL.DiffusionWYK.NUM_TEST_PROPOSALS = 100
    cfg.MODEL.DiffusionWYK.KNOWN_NOISE_LEVEL = 0.0
    cfg.MODEL.DiffusionWYK.ETA = 1.0

    # RCNN Head.
    cfg.MODEL.DiffusionWYK.NHEADS = 8
    cfg.MODEL.DiffusionWYK.DROPOUT = 0.0
    cfg.MODEL.DiffusionWYK.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionWYK.ACTIVATION = "relu"
    cfg.MODEL.DiffusionWYK.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionWYK.NUM_CLS = 1
    cfg.MODEL.DiffusionWYK.NUM_REG = 3
    cfg.MODEL.DiffusionWYK.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionWYK.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionWYK.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionWYK.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionWYK.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionWYK.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionWYK.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionWYK.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionWYK.USE_FOCAL = True
    cfg.MODEL.DiffusionWYK.USE_FED_LOSS = False
    cfg.MODEL.DiffusionWYK.ALPHA = 0.25
    cfg.MODEL.DiffusionWYK.GAMMA = 2.0
    cfg.MODEL.DiffusionWYK.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionWYK.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionWYK.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionWYK.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionWYK.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = "B"  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (
        400,
        500,
        600,
        640,
        700,
        900,
        1000,
        1100,
        1200,
        1300,
        1400,
        1800,
        800,
    )
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = (
        [96, 10000],
        [96, 10000],
        [64, 10000],
        [64, 10000],
        [64, 10000],
        [0, 10000],
        [0, 10000],
        [0, 256],
        [0, 256],
        [0, 192],
        [0, 192],
        [0, 96],
        [0, 10000],
    )
