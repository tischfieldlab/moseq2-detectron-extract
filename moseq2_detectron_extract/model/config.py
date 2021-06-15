from moseq2_detectron_extract.io.annot import get_dataset_statistics
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog



def get_base_config():
    cfg = get_cfg()


    # USE POINTREND
    #point_rend.add_pointrend_config(cfg)
    #cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")

    # USE Keypoint RCNN
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")

    # USE MASK RCNN
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 24   # faster, and good enough for this toy dataset (default: 512)

    cfg.MODEL.KEYPOINT_ON = True

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500


    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.INPUT.FORMAT = "L"
    cfg.INPUT.MIN_SIZE_TRAIN = (240,)
    cfg.INPUT.MAX_SIZE_TRAIN = 250
    cfg.INPUT.MIN_SIZE_TEST = 240
    cfg.INPUT.MAX_SIZE_TEST = 250
    cfg.INPUT.RANDOM_FLIP = "none"


    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = (2500,) #(5000, 10000, 15000)        # do not decay learning rate


    cfg.OUTPUT_DIR = './output_4'
    cfg.VIS_PERIOD = 100
    #cfg.MODEL.WEIGHTS = None

    cfg.TEST.DETECTIONS_PER_IMAGE = 1

    return cfg


def add_dataset_cfg(cfg, train_dset_name="moseq_train", test_dset_name="moseq_test", recompute_pixel_stats=False):
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
        cfg.MODEL.PIXEL_MEAN = [float(px_mean)]
        cfg.MODEL.PIXEL_STD = [float(px_stdev)]
    else:
        # use premeasured
        cfg.MODEL.PIXEL_MEAN = [1.8554014629469981]
        cfg.MODEL.PIXEL_STD = [6.392353752797691]


    return cfg
