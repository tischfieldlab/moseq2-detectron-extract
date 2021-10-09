from concurrent.futures.thread import ThreadPoolExecutor
from moseq2_detectron_extract.io.proc import apply_roi, instances_to_features
import os

from detectron2.data.catalog import MetadataCatalog
import tqdm
from moseq2_detectron_extract.model.model import Predictor
from moseq2_detectron_extract.io.session import Session
from queue import Queue



def extract(session: Session, predictor: Predictor, output_dir, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
            bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, min_height, max_height, 
            chunk_size, chunk_overlap, frame_dtype):
    # set up the output directory
    if output_dir is None:
        output_dir = os.path.join(session.dirname, 'proc')
    else:
        output_dir = os.path.join(session.dirname, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    info_dir = os.path.join(images_dir, '.info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)

    # Find image background and ROI
    bground_im, roi, true_depth = session.find_roi(bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
            bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, cache_dir=info_dir)
    print(f'Found true depth: {true_depth}')



    q = Queue()
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:

        kp_out_data = []
        keypoint_names = MetadataCatalog.get("moseq_train").keypoint_names
        for frame_idxs, raw_frames in tqdm.tqdm(session.iterate(chunk_size, chunk_overlap), desc='Processing batches'):
            raw_frames = bground_im - raw_frames
            raw_frames[raw_frames < min_height] = 0
            raw_frames[raw_frames > max_height] = max_height
            raw_frames = (raw_frames / max_height) * 255 # rescale to use full gammitt
            raw_frames = raw_frames.astype(frame_dtype)
            raw_frames = apply_roi(raw_frames, roi)


def do_inference(predictor, raw_frames, batch_size):
    # Do the inference
    outputs = []
    for i in tqdm.tqdm(range(0, raw_frames.shape[0], batch_size), desc="Inferring", leave=False):
        #plt.imshow((raw_frames[i,:,:] / max_height) * 255)
        #plt.show()
        outputs.extend(predictor(raw_frames[i:i+batch_size,:,:,None])) # rescale to use full gammitt

    angles, centroids, masks, flips, allosteric_keypoints, rotated_keypoints = instances_to_features(outputs, raw_frames)