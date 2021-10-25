import numpy as np
import tqdm
from moseq2_detectron_extract.io.proc import apply_roi
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans


def select_frames_kmeans(session, num_frames_to_pick, num_clusters=None, chunk_size=1000, scale=4, min_height=5, max_height=100, kmeans_batchsize=100, kmeans_max_iter=50):

    first_frame, bground_im, roi, true_depth = session.find_roi()
    downsampled = np.zeros((session.nframes, int(roi.shape[0] / scale), int(roi.shape[1] / scale)))
    for frame_idxs, raw_frames in tqdm.tqdm(session.iterate(chunk_size=chunk_size), desc='Processing batches', leave=False):
        raw_frames = bground_im - raw_frames
        raw_frames[raw_frames < min_height] = 0
        raw_frames[raw_frames > max_height] = max_height
        raw_frames = apply_roi(raw_frames, roi)
        raw_frames = raw_frames
        
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
