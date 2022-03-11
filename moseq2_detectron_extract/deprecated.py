

# @cli.command(name='infer', help='run inference')
# @click.argument('model_dir', nargs=1, type=click.Path(exists=True))
# @click.argument('input_file', nargs=1, type=click.Path(exists=True))
# @click.option('--checkpoint', default='last', help='Model checkpoint to load. Use "last" to load the last checkpoint.')
# @click.option('--frame-trim', default=(0, 0), type=(int, int), help='Frames to trim from beginning and end of data')
# @click.option('--batch-size', default=10, type=int, help='Number of frames for each model inference iteration')
# @click.option('--chunk-size', default=1000, type=int, help='Number of frames for each processing iteration')
# @click.option('--chunk-overlap', default=0, type=int, help='Frames overlapped in each chunk. Useful for cable tracking')
# @click.option('--bg-roi-dilate', default=(10, 10), type=(int, int), help='Size of the mask dilation (to include environment walls)')
# @click.option('--bg-roi-shape', default='ellipse', type=str, help='Shape to use for the mask dilation (ellipse or rect)')
# @click.option('--bg-roi-index', default=0, type=int, help='Index of which background mask(s) to use')
# @click.option('--bg-roi-weights', default=(1, .1, 1), type=(float, float, float), help='Feature weighting (area, extent, dist) of the background mask')
# @click.option('--bg-roi-depth-range', default=(650, 750), type=(float, float), help='Range to search for floor of arena (in mm)')
# @click.option('--bg-roi-gradient-filter', default=False, type=bool, help='Exclude walls with gradient filtering')
# @click.option('--bg-roi-gradient-threshold', default=3000, type=float, help='Gradient must be < this to include points')
# @click.option('--bg-roi-gradient-kernel', default=7, type=int, help='Kernel size for Sobel gradient filtering')
# @click.option('--bg-roi-fill-holes', default=True, type=bool, help='Fill holes in ROI')
# @click.option('--use-plane-bground', is_flag=True, help='Use a plane fit for the background. Useful for mice that don\'t move much')
# @click.option('--frame-dtype', default='uint8', type=click.Choice(['uint8', 'uint16']), help='Data type for processed frames')
# @click.option('--output-dir', default=None, help='Output directory to save the results h5 file')
# @click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
# @click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
# @click.option('--fps', default=30, type=int, help='Frame rate of camera')
# @click.option('--crop-size', default=(80, 80), type=(int, int), help='size of crop region')
# @click.option("--profile", is_flag=True)
# def infer(model_dir, input_file, checkpoint, frame_trim, batch_size, chunk_size, chunk_overlap, bg_roi_dilate, bg_roi_shape, bg_roi_index,
#           bg_roi_weights, bg_roi_depth_range, bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes,
#           use_plane_bground, frame_dtype, output_dir, min_height, max_height, fps, crop_size, profile):
#     logging.info("") # Empty line to give some breething room

#     if profile:
#         enable_profiling()

#     config_data = locals()
#     config_data.update({
#         'use_tracking_model': False,
#         'flip_classifier': model_dir,
#     })

#     status_dict = {
#         'complete': False,
#         'skip': False,
#         'uuid': str(uuid.uuid4()),
#         'metadata': '',
#         'parameters': deepcopy(config_data)
#     }

#     session = Session(input_file, frame_trim=frame_trim)

#     # set up the output directory
#     if output_dir is None:
#         output_dir = os.path.join(session.dirname, 'proc')
#     else:
#         output_dir = os.path.join(session.dirname, output_dir)

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     tee = Tee(os.path.join(output_dir, 'infer.log'))
#     tee.attach()

#     info_dir = os.path.join(output_dir, '.info')
#     if not os.path.exists(info_dir):
#         os.makedirs(info_dir)

#     logging.info('Loading model....')
#     register_dataset_metadata("moseq_train", default_keypoint_names)
#     cfg = get_base_config()
#     with open(os.path.join(model_dir, 'config.yaml'), 'r', encoding='utf-8') as cfg_file:
#         cfg = cfg.load_cfg(cfg_file)
#     if checkpoint == 'last':
#         logging.info(' -> Using last model checkpoint....')
#         cfg.MODEL.WEIGHTS = get_last_checkpoint(model_dir)
#     else:
#         logging.info(f' -> Using model checkpoint at iteration {checkpoint}....')
#         cfg.MODEL.WEIGHTS = get_specific_checkpoint(model_dir, checkpoint)
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
#     cfg.TEST.DETECTIONS_PER_IMAGE = 1
#     predictor = Predictor(cfg)

#     logging.info(f'Processing: {input_file}')
#     # Find image background and ROI
#     first_frame, bground_im, roi, true_depth = session.find_roi(bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
#             bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, cache_dir=info_dir)
#     logging.info(f'Found true depth: {true_depth}')
#     config_data.update({
#         'true_depth': true_depth,
#     })

#     preview_video_dest = os.path.join(output_dir, 'extraction.mp4')
#     video_pipe = PreviewVideoWriter(preview_video_dest, fps=fps, vmin=min_height, vmax=max_height)

#     result_h5_dest = os.path.join(output_dir, 'result.h5')
#     result_h5 = h5py.File(result_h5_dest, mode='w')
#     create_extract_h5(result_h5, acquisition_metadata=session.load_metadata(), config_data=config_data, status_dict=status_dict,
#                       nframes=session.nframes, roi=roi, bground_im=bground_im, first_frame=first_frame, timestamps=session.load_timestamps(Stream.DEPTH))

#     times = {
#         'prepare_data': [],
#         'inference': [],
#         'features': [],
#         'draw_instances': [],
#         'write_keypoints': [],
#         'crop_rotate': [],
#         'colorize': [],
#         'write_video': []
#     }

#     # Iterate Frames and write images
#     last_frame = None
#     kp_out_data = []
#     keypoint_names = MetadataCatalog.get("moseq_train").keypoint_names
#     for i, (frame_idxs, raw_frames) in enumerate(tqdm.tqdm(session.iterate(chunk_size, chunk_overlap), desc='Processing batches')):
#         offset = chunk_overlap if i > 0 else 0
#         start = time.time()
#         raw_frames = prep_raw_frames(raw_frames, bground_im=bground_im, roi=roi, vmin=min_height, vmax=max_height)
#         times['prepare_data'].append(time.time() - start)



#         # Do the inference
#         start = time.time()
#         outputs = []
#         for i in tqdm.tqdm(range(0, raw_frames.shape[0], batch_size), desc="Inferring", leave=False):
#             outputs.extend(predictor(scale_raw_frames(raw_frames[i:i+batch_size,:,:,None], vmin=min_height, vmax=max_height)))
#         times['inference'].append(time.time() - start)

#         # Post process results and extract features
#         start = time.time()
#         features = instances_to_features(outputs, raw_frames)
#         times['features'].append(time.time() - start)


#         sub_times = {
#             'draw_instances': [],
#             'write_keypoints': [],
#             'crop_rotate': [],
#         }
#         rfs = raw_frames.shape
#         scale = 2.0
#         out_video = np.zeros((rfs[0], int(rfs[1]*scale), int(rfs[2]*scale), 3), dtype='uint8')
#         cropped_frames = np.zeros((rfs[0], crop_size[0], crop_size[1]), dtype='uint8')
#         cropped_masks = np.zeros((rfs[0], crop_size[0], crop_size[1]), dtype='uint8')
#         for i in tqdm.tqdm(range(raw_frames.shape[0]), desc="Postprocessing", leave=False):
#             raw_frame = raw_frames[i]
#             clean_frame = features['cleaned_frames'][i]
#             mask = features['masks'][i]
#             output = outputs[i]
#             angle = features['features']['orientation'][i]
#             centroid = features['features']['centroid'][i]
#             flip = features['flips'][i]
#             allocentric_keypoints = features['allocentric_keypoints'][i, 0]
#             rotated_keypoints = features['rotated_keypoints'][i, 0]



#             if len(instances) <= 0:
#                 tqdm.tqdm.write(f"WARNING: No instances found for frame #{frame_idxs[i]}")

#             start = time.time()
#             kp_out_data.append({
#                 'Frame_Idx': frame_idxs[i],
#                 'Flip': flip,
#                 'Centroid_X': centroid[0],
#                 'Centroid_Y': centroid[1],
#                 'Angle': angle,
#                 **keypoints_to_dict(keypoint_names, allocentric_keypoints),
#                 **keypoints_to_dict(keypoint_names, rotated_keypoints, prefix='rot_')
#             })
#             sub_times['write_keypoints'].append(time.time() - start)

#             instances = output["instances"].to('cpu')
#             start = time.time()
#             out_video[i,:,:,:] = draw_instances_fast(raw_frame[:,:,None].copy(), instances, scale=scale)
#             sub_times['draw_instances'].append(time.time() - start)

#             start = time.time()
#             cropped = crop_and_rotate_frame(raw_frame, centroid, angle, crop_size)
#             cropped_mask = crop_and_rotate_frame(mask, centroid, angle, crop_size)
#             cropped = cropped * cropped_mask # mask the cropped image
#             cropped_frames[i] = cropped
#             cropped_masks[i] = cropped_mask
#             sub_times['crop_rotate'].append(time.time() - start)

#         results = {
#             'chunk': raw_frames,
#             'depth_frames': cropped_frames,
#             'mask_frames': cropped_masks,
#             'scalars': compute_scalars(raw_frames * features['masks'],
#                                        features['features'],
#                                        min_height=min_height,
#                                        max_height=max_height,
#                                        true_depth=true_depth),
#             'flips': features['flips'],
#             'parameters': None # only not None if EM tracking was used (we don't support that here)
#         }

#         write_extracted_chunk_to_h5(result_h5, results=results, frame_range=frame_idxs, offset=offset)



#         start = time.time()
#         out_video_combined = stack_videos([out_video, colorize_video(cropped_frames, vmax=255)], orientation='diagional')
#         video_pipe.write_frames(frame_idxs, out_video_combined)
#         times['write_video'].append(time.time() - start)

#         for k, v in sub_times.items():
#             times[k].append(np.sum(v))

#     pd.DataFrame(kp_out_data).to_csv(os.path.join(output_dir, 'keypoints.tsv'), sep='\t', index=False)
#     result_h5.close()
#     video_pipe.close()

#     logging.info('Processing Times:')
#     for k, v in times.items():
#         logging.info(f'{k}: {np.sum(v)}')
#     logging.info(f'Total: {np.sum(list(times.values()))}')
