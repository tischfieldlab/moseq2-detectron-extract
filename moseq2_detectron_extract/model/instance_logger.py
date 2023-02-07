from detectron2.structures import Instances, pairwise_iou
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cdist

class InstanceLogger():
    '''Class to log detectron2 instances'''

    def __init__(self, dest: str) -> None:
        self.dest = dest
        self.file = open(self.dest, 'w+', encoding="utf-8")
        self._write_headers()

    def close(self):
        '''Close this logger'''
        if self.file:
            self.file.close()

    def _write_headers(self):
        '''Write headers to the log'''
        headers = [
            "frame_idx",
            "box_jaccard",
            "mask_jaccard",
            "kp_dist"
        ]
        self.file.write("\t".join(headers) + "\n")

    def log_instances(self, frame_idx: int, instances: Instances):
        '''Write a set of instances to this log'''
        self.file.write(f"{frame_idx}\t")
        if len(instances) > 1:
            self.file.write(f"{pairwise_iou(instances.pred_boxes[0], instances.pred_boxes[1])[0, 0]}\t")
            self.file.write(f"{jaccard_score(instances.pred_masks[0], instances.pred_masks[1], average='micro')}\t")
            self.file.write(f"{cdist(instances.pred_keypoints[0, :8, :2], instances.pred_keypoints[1, :8, :2]).diagonal().sum()}\t")

        else:
            self.file.write("0\t0\t")

        for i in range(len(instances)):
            self.file.write(f"{instances.scores[i]}\t")
        self.file.write("\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
