import os
import tarfile
from moseq2_unet_extract.io.image import read_image, write_image
from moseq2_unet_extract.io.util import (gen_batch_sequence, load_metadata,
                                         load_timestamps)
from moseq2_unet_extract.io.video import get_movie_info, load_movie_data
from moseq2_unet_extract.proc.util import (apply_roi, get_bground_im_file,
                                           get_roi, select_strel)
import numpy as np

class Session(object):

    def __init__(self, path, frame_trim=(0, 0)):
        self.init_session(path)
        self.trim_frames(frame_trim)


    def init_session(self, input_file):
        self.dirname = os.path.dirname(input_file)

        if input_file.endswith('.tar.gz') or input_file.endswith('.tgz'):
            print('Scanning tarball {} (this will take a minute)'.format(input_file))
            #compute NEW psuedo-dirname now, `input_file` gets overwritten below with depth.dat tarinfo...
            self.dirname = os.path.join(self.dirname, os.path.basename(input_file).replace('.tar.gz', '').replace('.tgz', ''))

            self.tar = tarfile.open(input_file, 'r:gz')
            self.tar_members = self.tar.getmembers()
            self.tar_names = [_.name for _ in self.tar_members]
            self.input_file = self.tar_members[self.tar_names.index('depth.dat')]
            self.rgb_file = self.tar_members[self.tar_names.index('rgb.mp4')]
            self.session_id = os.path.basename(input_file).split('.')[0]
        else:
            self.tar = None
            self.tar_members = None
            self.input_file = input_file
            self.rgb_file = os.path.join(self.dirname, 'rgb.mp4')
            self.session_id = os.path.basename(os.path.dirname(input_file))

        self.video_metadata = get_movie_info(self.input_file)
    #end init_session()

    # def prep_directory_structure(self):
    #     pass
    # #end prep_directory_structure()

    def trim_frames(self, frame_trim):
        self.nframes = self.video_metadata['nframes']

        if frame_trim[0] > 0 and frame_trim[0] < self.nframes:
            self.first_frame_idx = frame_trim[0]
        else:
            self.first_frame_idx = 0

        if self.nframes - frame_trim[1] > self.first_frame_idx:
            self.last_frame_idx = self.nframes - frame_trim[1]
        else:
            self.last_frame_idx = self.nframes

        self.nframes = self.last_frame_idx - self.first_frame_idx
    #end trim_frames()

    def load_metadata(self):
        if self.tar is not None:
            metadata_path = self.tar.extractfile(self.tar_members[self.tar_names.index('metadata.json')])
        else:
            metadata_path = os.path.join(self.dirname, 'metadata.json')
        return load_metadata(metadata_path)
    #end load_metadata()

    def load_timestamps(self):
        correction_factor = 1.0

        if self.tar is not None:
            if "depth_ts.txt" in self.tar_names:
                timestamp_path = self.tar.extractfile(self.tar_members[self.tar_names.index('depth_ts.txt')])
            elif "timestamps.csv" in self.tar_names:
                timestamp_path = self.tar.extractfile(self.tar_members[self.tar_names.index('timestamps.csv')])
                correction_factor = 1000.0
        else:
            timestamp_path = os.path.join(self.dirname, 'depth_ts.txt')
            alternate_timestamp_path = os.path.join(self.dirname, 'timestamps.csv')
            if not os.path.exists(timestamp_path) and os.path.exists(alternate_timestamp_path):
                timestamp_path = alternate_timestamp_path
                correction_factor = 1000.0

        timestamps = load_timestamps(timestamp_path, col=0)

        if timestamps is not None:
            timestamps = timestamps[self.first_frame_idx:self.last_frame_idx]

        timestamps *= correction_factor

        return timestamps
    #end load_timestamps()


    def find_roi(self, bg_roi_dilate=(10,10), bg_roi_shape='ellipse', bg_roi_index=0, bg_roi_weights=(1, .1, 1),
                 bg_roi_depth_range=(650, 750), bg_roi_gradient_filter=False, bg_roi_gradient_threshold=3000,
                 bg_roi_gradient_kernel=7, bg_roi_fill_holes=True, use_plane_bground=False, cache_dir=None):

        if cache_dir and os.path.exists(os.path.join(cache_dir, 'bground.tiff')):
            print('Loading background...')
            bground_im = read_image(os.path.join(cache_dir, 'bground.tiff'), scale=True)
        else:
            print('Getting background...')
            bground_im = get_bground_im_file(self.input_file, tar_object=self.tar)

            if cache_dir and not use_plane_bground:
                write_image(os.path.join(cache_dir, 'bground.tiff'), bground_im, scale=True)

        if cache_dir:
            first_frame = load_movie_data(self.input_file, 0, tar_object=self.tar)
            write_image(os.path.join(cache_dir, 'first_frame.tiff'), first_frame[0], scale=True, scale_factor=bg_roi_depth_range)

        strel_dilate = select_strel(bg_roi_shape, bg_roi_dilate)

        roi_filename = 'roi_{:02d}.tiff'.format(bg_roi_index)

        if cache_dir and os.path.exists(os.path.join(cache_dir, roi_filename)):
            print('Loading ROI...')
            roi = read_image(os.path.join(cache_dir, roi_filename), scale=True) > 0
        else:
            print('Getting roi...')
            rois, plane, _, _, _, _ = get_roi(bground_im,
                                            strel_dilate=strel_dilate,
                                            weights=bg_roi_weights,
                                            depth_range=bg_roi_depth_range,
                                            gradient_filter=bg_roi_gradient_filter,
                                            gradient_threshold=bg_roi_gradient_threshold,
                                            gradient_kernel=bg_roi_gradient_kernel,
                                            fill_holes=bg_roi_fill_holes)

            if use_plane_bground:
                print('Using plane fit for background...')
                xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
                coords = np.vstack((xx.ravel(), yy.ravel()))
                plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
                plane_im = plane_im.reshape(bground_im.shape)
                if cache_dir:
                    write_image(os.path.join(cache_dir, 'bground.tiff'), plane_im, scale=True)
                bground_im = plane_im

            roi = rois[bg_roi_index]
            if cache_dir:
                write_image(os.path.join(cache_dir, roi_filename), roi, scale=True, dtype='uint8')

        true_depth = np.median(bground_im[roi > 0])
        print('Detected true depth: {}'.format(true_depth))

        return bground_im, roi, true_depth
    #end find_roi()

    def iterate(self, chunk_size=1000, chunk_overlap=0):
        return SessionFramesIterator(self, chunk_size, chunk_overlap)

    def sample(self, num_samples, chunk_size=1000):
        return SessionFramesSampler(self, num_samples, chunk_size, 0)

#end class Session


class SessionFramesIterator(object):
    def __init__(self, session, chunk_size, chunk_overlap):
        self.session = session
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batches = list(self.generate_samples())
        self.current = 0

    def generate_samples(self):
        return gen_batch_sequence(self.session.nframes, self.chunk_size, self.chunk_overlap, self.session.first_frame_idx)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        else:
            frame_range = self.batches[self.current]
        self.current += 1

        frame_idxs = list(frame_range)
        raw_frames = load_movie_data(self.session.input_file, frame_idxs, tar_object=self.session.tar)

        return frame_idxs, raw_frames


class SessionFramesSampler(SessionFramesIterator):
    def __init__(self, session, num_samples, chunk_size, chunk_overlap):
        self.num_samples = int(num_samples)
        super().__init__(session, chunk_size, chunk_overlap)

    def generate_samples(self):
        """Generate a sequence with overlap
        """
        offset = self.session.first_frame_idx
        seq = range(offset, self.session.nframes)
        seq = np.random.choice(seq, self.num_samples, replace=False)
        for i in range(offset, len(seq)-self.chunk_overlap, self.chunk_size-self.chunk_overlap):
            yield seq[i:i+self.chunk_size]
