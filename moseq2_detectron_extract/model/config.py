from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from moseq2_detectron_extract.io.annot import get_dataset_statistics


def load_config(config_file: str) -> CfgNode:
    ''' Load a configuration file

    Parameters:
    config_file (str): path to configuration file

    Returns:
    CfgNode: parsed configuration
    '''
    with open(config_file, 'r', encoding='utf-8') as cfg_file:
        config = CfgNode.load_cfg(cfg_file)
    return config


def get_base_config() -> CfgNode:
    ''' Get the base configuration

    Returns:
    CfgNode: base configuration
    '''
    cfg = get_cfg()

    # USE Keypoint RCNN
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.KEYPOINT_ON = True

    # Turn on mask detection
    cfg.MODEL.MASK_ON = True

    # We only have one class
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.5

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500

    # 4 workers for the data loader
    cfg.DATALOADER.NUM_WORKERS = 4

    # Some information about the input images
    cfg.INPUT.FORMAT = "L"
    cfg.INPUT.MIN_SIZE_TRAIN = (240,)
    cfg.INPUT.MAX_SIZE_TRAIN = 250
    cfg.INPUT.MIN_SIZE_TEST = 240
    cfg.INPUT.MAX_SIZE_TEST = 250
    cfg.INPUT.RANDOM_FLIP = "none"


    #some configuration for the solver
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    #cfg.SOLVER.MAX_ITER = 100000#50000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    #cfg.SOLVER.STEPS = (75000, 85000, 45000)        # do not decay learning rate
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = (70000, 80000, 90000)
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.AMP.ENABLED = True


    cfg.OUTPUT_DIR = './models/output_4'
    cfg.VIS_PERIOD = 100
    cfg.CUDNN_BENCHMARK = True

    cfg.TEST.DETECTIONS_PER_IMAGE = 1
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.TEST.AUG.FLIP = False


    # emperically tuned parameters
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.FPN.NORM = 'GN'
    cfg.MODEL.FPN.FUSE_TYPE = 'avg'
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5

    return cfg


def add_dataset_cfg(cfg: CfgNode, train_dset_name: str="moseq_train", test_dset_name: str="moseq_test", recompute_pixel_stats: bool=True) -> CfgNode:
    ''' Add dataset-specific configuration details to the config

    Parameters:
    cfg (CfgNode): configuration to update
    train_dset_name (str): name of the training dataset
    test_dset_name (str): name of the testing dataset
    recompute_pixel_stats (bool): True to recompute pixel statistics

    Returns:
    CfgNode: updated configuration
    '''
    cfg.DATASETS.TRAIN = (train_dset_name,)
    cfg.DATASETS.TEST = (test_dset_name,)

    metadata = MetadataCatalog.get(train_dset_name)
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(metadata.keypoint_names)
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [
        0.026, # Nose
        0.035, # Left Ear
        0.035, # Right Ear
        0.079, # Neck
        0.107, # Left Hip
        0.107, # Right Hip
        0.089, # TailBase
        0.026, # TailTip
    ]

    if recompute_pixel_stats:
        px_mean, px_stdev = get_dataset_statistics(DatasetCatalog.get(train_dset_name))
        cfg.MODEL.PIXEL_MEAN = [float(pm) for pm in px_mean]
        cfg.MODEL.PIXEL_STD = [float(ps) for ps in px_stdev]
    else:
        # use premeasured
        cfg.MODEL.PIXEL_MEAN = [1.8554014629469981]
        cfg.MODEL.PIXEL_STD = [6.392353752797691]

    return cfg
