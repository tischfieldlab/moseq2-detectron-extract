from typing import List
import numpy as np
import tqdm
from moseq2_detectron_extract.io.session import Session
from moseq2_detectron_extract.proc.proc import (prep_raw_frames,
                                                scale_raw_frames)
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans


def select_frames_kmeans(session: Session, num_frames_to_pick: int, num_clusters: int=None, chunk_size: int=1000,
    scale: float=4, min_height: int=0, max_height: int=100, kmeans_batchsize: int=100, kmeans_max_iter: int=50) -> List[int]:
    ''' Select frames from a session using k-means clustering to pick dissimilar frames

    Parameters:
    session (Session): Session from which to pick frames
    num_frames_to_pick (int): Total number of frames to pick from this session
    num_clusters (int): Number of (k-means) clusters to use. If none, then `num_frames_to_pick` clusters are used
    chunk_size (int): Number of frames to read from session in an iteration
    scale (float): Down-sample images by this scale factor
    min_height (int): Min height of the animal
    max_height (int): Max height of the animal
    kmeans_batchsize (int): Batch size for fitting k-means model
    kmeans_max_iter (int): Max iterations for fitting k-means model

    Returns:
    list of frame indicies which were selected
    '''

    _, bground_im, roi, _ = session.find_roi()
    downsampled = np.zeros((session.nframes, int(roi.shape[0] / scale), int(roi.shape[1] / scale)))
    for frame_idxs, raw_frames in tqdm.tqdm(session.iterate(chunk_size=chunk_size), desc='Processing batches', leave=False):
        raw_frames = prep_raw_frames(raw_frames, bground_im=bground_im, roi=roi, vmin=min_height, vmax=max_height)
        raw_frames = scale_raw_frames(raw_frames, vmin=min_height, vmax=max_height)

        for i, idx in enumerate(tqdm.tqdm(frame_idxs, desc='Resizing Frames', leave=False, disable=False)):
            downsampled[idx, :, :] = resize(raw_frames[i], downsampled.shape[1:], anti_aliasing=True, preserve_range=True, mode='constant')

    with tqdm.tqdm(total=1, leave=False, desc="Kmeans clustering ... (this might take a while)") as pbar:
        data = downsampled - downsampled.mean(axis=0)
        data = data.reshape(data.shape[0], -1)  # stacking

        if num_clusters is None:
            num_clusters = num_frames_to_pick

        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            tol=1e-3,
            batch_size=kmeans_batchsize,
            max_iter=kmeans_max_iter
        )
        kmeans.fit(data)
        pbar.update(1)

    num_frames_per_cluster = num_frames_to_pick // num_clusters
    if num_frames_per_cluster < 1:
        num_frames_per_cluster = 1

    selected_frames = []
    for cluster_id in range(num_clusters):  # pick one frame per cluster
        cluster_ids = np.where(cluster_id == kmeans.labels_)[0]

        num_images_in_cluster = len(cluster_ids)
        if num_images_in_cluster > 0:
            selected_frames.extend(list(cluster_ids[np.random.choice(num_images_in_cluster, size=num_frames_per_cluster, replace=False)]))

    return selected_frames
