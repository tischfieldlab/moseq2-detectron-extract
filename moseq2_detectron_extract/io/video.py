import datetime
import os
import subprocess
import tarfile
import tempfile
from copy import Error
from itertools import groupby
from operator import itemgetter
from typing import Iterable, List, Tuple, TypedDict, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm


class RawVideoInfo(TypedDict):
    bytes: int
    nframes: int
    dims: Tuple[int, int]
    bytes_per_frame: int


def get_raw_info(filename: str, bit_depth: int=16, frame_dims: Tuple[int, int]=(512, 424)) -> RawVideoInfo:
    ''' Get info from a raw data file with specified frame dimensions and bit depth

    Parameters:
    filename (string): name of raw data file
    bit_depth (int): bits per pixel (default: 16)
    frame_dims (tuple): wxh or hxw of each frame

    Returns:
    RawVideoInfo, dict containg information about the raw data file
    '''

    bytes_per_frame = (frame_dims[0] * frame_dims[1] * bit_depth) / 8

    if type(filename) is str:
        file_info = {
            'bytes': os.stat(filename).st_size,
            'nframes': int(os.stat(filename).st_size / bytes_per_frame),
            'dims': frame_dims,
            'bytes_per_frame': bytes_per_frame
        }
    elif type(filename) is tarfile.TarInfo:
        file_info = {
            'bytes': filename.size,
            'nframes': int(filename.size / bytes_per_frame),
            'dims': frame_dims,
            'bytes_per_frame': bytes_per_frame
        }
    return file_info

FramesSelection = Union[int, Iterable[int]]
def read_frames_raw(filename: str, frames: FramesSelection=None, frame_dims: Tuple[int, int]=(512, 424), bit_depth: int=16,
    dtype: npt.DTypeLike="<i2", tar_object: tarfile.TarInfo=None) -> np.ndarray:
    '''
    Reads in data from raw binary file

    Args:
        filename (string): name of raw data file
        frames (Union[int, Iterable[int]]): frame indicies to read
        frame_dims (tuple): (width, height) of frames in pixels
        bit_depth (int): bits per pixel (default: 16)
        tar_object (tarfile.TarFile): TarFile object, used for loading data directly from tgz

    Returns:
        a numpy.ndarray of frames with shape (nframes, height, width)
    '''

    vid_info = get_raw_info(filename, frame_dims=frame_dims, bit_depth=bit_depth)

    if isinstance(frames, int):
        # single frame index, we will just return that one frame
        frames = [frames]
    elif isinstance(frames, Iterable):
        # an iterable of indicies, ensure they are all ints
        frames = [int(i) for i in frames]

    # sanity check on `frames`, and allow passing of None
    if frames is None or len(frames) == 0:
        frames = list(range(0, vid_info['nframes']))

    # Build blocks of consecutive frames we can read as one batch.
    # Accounts for possible non-monotonically increasing indicies, and random access cases
    blocks = []
    for start, nframes in collapse_consecutive_values(sorted(frames)):
        blocks.append({
            'seek_point': int(np.maximum(0, start * vid_info['bytes_per_frame'])),
            'read_bytes': int(nframes * vid_info['bytes_per_frame']),
            'read_points': int(nframes * frame_dims[0] * frame_dims[1]),
            'dims': (nframes, frame_dims[1], frame_dims[0]),
            'idxs': [frames.index(start + i) for i in range(nframes)]
        })

    out_buffer = np.empty((len(frames), frame_dims[1], frame_dims[0]), dtype=np.dtype(dtype))
    if type(tar_object) is tarfile.TarFile:
        with tar_object.extractfile(filename) as f:
            for b in blocks:
                f.seek(b['seek_point'])
                chunk = f.read(b['read_bytes'])
                chunk = np.frombuffer(chunk, dtype=np.dtype(dtype)).reshape(b['dims'])
                out_buffer[b['idxs'], ...] = chunk
    else:
        with open(filename, "rb") as f:
            for b in blocks:
                f.seek(b['seek_point'])
                chunk = np.fromfile(file=f, dtype=np.dtype(dtype), count=b['read_points']).reshape(b['dims'])
                out_buffer[b['idxs'], ...] = chunk

    return out_buffer


def collapse_consecutive_values(a: np.ndarray) -> List[Tuple[float, int]]:
    ''' Collapses consecutive values in an array

    Example:
    collapse_consecutive_values([0,1,2,3,10,11,12,13,21,22,23])
    > [(0,4), (10,4), (21, 3)]

    Parameters:
    a (np.ndarray): array of labels to collapse

    Returns
    List[Tuple[float, int]]: each tuple contains (seed, run_count)
    '''
    grouped_instances = []
    for _, g in groupby(enumerate(a), lambda ix : ix[0] - ix[1]):
        local_group = list(map(itemgetter(1), g))
        grouped_instances.append((local_group[0], len(local_group)))
    return grouped_instances
#end collapse_adjacent_values()


class FFProbeInfo(TypedDict):
    file: str
    codec: str
    pixel_format: str
    dims: Tuple[int, int]
    fps: float
    nframes: int


# https://gist.github.com/hiwonjoon/035a1ead72a767add4b87afe03d0dd7b
def get_video_info(filename: str, tar_object: tarfile.TarInfo=None) -> FFProbeInfo:
    '''
    Get dimensions of data compressed using ffv1, along with duration via ffmpeg

    Parameters:
    filename (str): name of file
    tar_object (tarfile.TarInfo|None): tarfile to read from. None if filename is not compressed

    Returns:
    FFProbeInfo - dict containing information about video `filename`
    '''
    is_stream = isinstance(filename, tarfile.TarInfo)
    if is_stream:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(tar_object.extractfile(filename).read())
        f.close()
        filename = f.name

    command = [
        'ffprobe',
        '-v', 'fatal',
        '-show_entries',
        'stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        filename,
        '-sexagesimal'
    ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()

    if is_stream:
        os.remove(f.name)

    if err:
        print(err)
    out = out.decode().split(os.linesep)

    return {
        'file': filename,
        'codec': out[0],
        'pixel_format': out[3],
        'dims': (int(out[1]), int(out[2])),
        'fps': float(out[4].split('/')[0])/float(out[4].split('/')[1]),
        'nframes': int(out[5])
    }


# simple command to pipe frames to an ffv1 file
def write_frames(filename: str, frames: np.ndarray, threads: int=6, fps: int=30,
                 pixel_format: str='gray16le', codec: str='ffv1', close_pipe: bool=True,
                 pipe=None, slices: int=24, slicecrc: int=1, frame_size: Tuple[int, int]=None, get_cmd=False):
    '''
    Write frames to avi file using the ffv1 lossless encoder
    '''

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = '{0:d}x{1:d}'.format(frames[0], frames[1])

    command = [
        'ffmpeg',
        '-y',
        '-loglevel', 'fatal',
        '-framerate', str(fps),
        '-f', 'rawvideo',
        '-s', frame_size,
        '-pix_fmt', pixel_format,
        '-i', '-',
        '-an',
        '-vcodec', codec,
        '-threads', str(threads),
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-r', str(fps),
        filename
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in tqdm.tqdm(range(frames.shape[0])):
        pipe.stdin.write(frames[i, ...].astype('uint16').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe


def read_frames(filename: str, frames=range(0,), threads: int=6, fps: int=30,
                pixel_format: str='gray16le', frame_size: Tuple[int, int]=None,
                slices: int=24, slicecrc: int=1, get_cmd=False, tar_object=None, **_):
    '''
    Reads in frames from the .nut/.avi file using a pipe from ffmpeg.

    Args:
        filename (str): filename to get frames from
        frames (list or 1d numpy array): list of frames to grab
        threads (int): number of threads to use for decode
        fps (int): frame rate of camera in Hz
        pixel_format (str): ffmpeg pixel format of data
        frame_size (str): wxh frame size in pixels
        slices (int): number of slices to use for decode
        slicecrc (int): check integrity of slices

    Returns:
        3d numpy array:  frames x h x w
    '''
    is_stream = isinstance(filename, tarfile.TarInfo)
    if is_stream:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(tar_object.extractfile(filename).read())
        f.close()
        filename = f.name

    finfo = get_video_info(filename)

    if frames is None or len(frames) == 0:
        frames = np.arange(finfo['nframes']).astype('int16')

    if not frame_size:
        frame_size = finfo['dims']

    if pixel_format == 'gray16le':
        dtype = 'uint16'
        out_shape = (len(frames), frame_size[1], frame_size[0])
    elif pixel_format == 'rgb24':
        dtype = 'uint8'
        out_shape = (len(frames), frame_size[1], frame_size[0], 3)

    out_video = np.empty(out_shape, dtype)
    for start, nframes in collapse_consecutive_values(sorted(frames)):
        command = [
            'ffmpeg',
            '-loglevel', 'fatal',
            '-ss', str(datetime.timedelta(seconds=start/fps)),
            '-i', filename,
            '-vframes', str(nframes),
            '-f', 'image2pipe',
            '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
            '-pix_fmt', pixel_format,
            '-threads', str(threads),
            '-slices', str(slices),
            '-slicecrc', str(slicecrc),
            '-vcodec', 'rawvideo',
            '-'
        ]

        pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = pipe.communicate()

        if(err):
            raise Error(err)

        idxs = [frames.index(start + i) for i in range(nframes)]
        out_video[idxs] = np.frombuffer(out, dtype=dtype).reshape((nframes, *out_shape[1:]))

    if is_stream:
        os.remove(f.name)

    return out_video
# simple command to pipe frames from an ffv1 file


def write_frames_preview(filename: str, frames=np.empty((0,)), threads: int=6,
                         fps: int=30, pixel_format: str='rgb24',
                         codec: str='h264', slices: int=24, slicecrc: int=1,
                         frame_size=None, depth_min: float=0, depth_max: float=80,
                         get_cmd: bool=False, cmap: str='jet',
                         pipe=None, close_pipe: bool=True, frame_range=None, tqdm_kwargs=None):
    '''
    Writes out a false-colored mp4 video
    '''

    if tqdm_kwargs is None:
        tqdm_kwargs = {}

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    txt_pos = (5, frames.shape[1] - 40)

    if not np.mod(frames.shape[1], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)

    if not np.mod(frames.shape[2], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=0)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = '{0:d}x{1:d}'.format(frames[0], frames[1])

    command = [
        'ffmpeg',
        '-y',
        '-loglevel', 'fatal',
        '-threads', str(threads),
        '-framerate', str(fps),
        '-f', 'rawvideo',
        '-s', frame_size,
        '-pix_fmt', pixel_format,
        '-i', '-',
        '-an',
        '-vcodec', codec,
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-r', str(fps),
        '-pix_fmt', 'yuv420p',
        filename
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # scale frames d00d

    use_cmap = plt.get_cmap(cmap)

    for i in tqdm.tqdm(range(frames.shape[0]), desc="Writing frames", **tqdm_kwargs):
        disp_img = frames[i, ...].copy().astype('float32')
        if len(frames.shape) == 3:
            # we have only single channel
            disp_img = (disp_img-depth_min)/(depth_max-depth_min)
            disp_img[disp_img < 0] = 0
            disp_img[disp_img > 1] = 1
            disp_img = np.delete(use_cmap(disp_img), 3, 2)*255

        if frame_range is not None:
            cv2.putText(disp_img, str(frame_range[i]), txt_pos, font, 1, white, 2, cv2.LINE_AA)
        pipe.stdin.write(disp_img.astype('uint8').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe


# def encode_raw_frames_chunk(src_filename, bground_im, roi, bbox,
#                             chunk_size=1000, overlap=0, depth_min=5,
#                             depth_max=100,
#                             bytes_per_frame=int((424*512*16)/8)):
#
#     save_dir = os.path.join(os.path.dirname(src_filename), '_chunks')
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     base_filename = os.path.splitext(os.path.basename(src_filename))[0]
#
#     file_bytes = os.stat(src_filename).st_size
#     file_nframes = int(file_bytes/bytes_per_frame)
#     steps = np.append(np.arange(0, file_nframes, chunk_size), file_nframes)
#
#     # need to write out a manifest so we know the location of every frame
#     dest_filename = []
#
#     for i in tqdm.tqdm(range(steps.shape[0]-1)):
#         if i == 1:
#             chunk = read_frames_raw(src_filename, np.arange(steps[i], steps[i+1]))
#         else:
#             chunk = read_frames_raw(src_filename, np.arange(steps[i]-overlap, steps[i+1]))
#
#         chunk = (bground_im-chunk).astype('uint8')
#         chunk[chunk < depth_min] = 0
#         chunk[chunk > depth_max] = 0
#         chunk = moseq2_extract.extract.proc.apply_roi(chunk, roi, bbox)
#
#         dest_filename.append(os.path.join(save_dir, base_filename+'chunk{:05d}.avi'.format(i)))
#         write_frames(dest_filename[-1], chunk)
#
#     return dest_filename


def load_movie_data(filename: str, frames=None, frame_dims: Tuple[int, int]=(512, 424), bit_depth: int=16, **kwargs):
    '''
    Reads in frames
    '''

    if isinstance(filename, tarfile.TarInfo):
        fname =  filename.name.lower()
    else:
        fname = filename.lower()

    if type(frames) is int:
        frames = [frames]

    if fname.endswith('.dat'):
        frame_data = read_frames_raw(filename,
                                     frames=frames,
                                     frame_dims=frame_dims,
                                     bit_depth=bit_depth,
                                     **kwargs)
    elif fname.endswith(('.avi', '.mp4')):
        frame_data = read_frames(filename, frames, **kwargs)

    return frame_data


def get_movie_info(filename: str, frame_dims: Tuple[int, int]=(512, 424), bit_depth: int=16, tar_object: tarfile.TarInfo=None):
    '''
    Gets movie info
    '''
    if isinstance(filename, tarfile.TarInfo):
        fname =  filename.name.lower()
    else:
        fname = filename.lower()

    if fname.endswith('.dat'):
        metadata = get_raw_info(filename, frame_dims=frame_dims, bit_depth=bit_depth)
    elif fname.endswith(('.avi', '.mp4')):
        metadata = get_video_info(filename, tar_object=tar_object)

    return metadata



class PreviewVideoWriter():
    def __init__(self, filename: str, fps: int=30, vmin: float=0, vmax: float=100, tqdm_opts: dict=None) -> None:
        self.filename = filename
        self.fps = fps
        self.vmin = vmin
        self.vmax = vmax
        self.video_pipe = None
        self.tqdm_opts = {
            'leave': False,
            'disable': True,
        }
        if tqdm_opts is not None:
            self.tqdm_opts.update(tqdm_opts)

    def write_frames(self, frame_idxs: np.ndarray, frames: np.ndarray):
        self.video_pipe = write_frames_preview(
                self.filename,
                frames,
                pipe=self.video_pipe, close_pipe=False,
                fps=self.fps,
                frame_range=frame_idxs,
                depth_max=self.vmin, depth_min=self.vmax,
                tqdm_kwargs=self.tqdm_opts)

    def close(self):
        if self.video_pipe:
            self.video_pipe.communicate()
