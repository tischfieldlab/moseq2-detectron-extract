import ast
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import tifffile
from imageio import imwrite


def write_image(filename: str, image: np.ndarray, scale: bool=True, scale_factor: Optional[Union[float, Tuple[float, float]]]=None,
                dtype: npt.DTypeLike='uint16', metadata: Optional[dict]=None, compress: int=0) -> None:
    ''' Save image data, possibly with scale factor for easy display

    Parameters:
    filename (str): path to file to write image
    image (np.ndarray): 2D numpy array containing image data
    scale (bool): If true, linear stretch image intensity values
    scale_factor (float|None): If not none, use scale image intensity by this amount, otherwise scale_factor is inferred from `dtype`
    dtype (npt.DTypeLike): dtype of the final image written to disk
    metadata: (dict|None): additional metadata to add to the image file
    compress (int): compression factor for the image file
    '''
    file = Path(filename)

    if metadata is None:
        metadata = {}

    if scale:
        max_int = np.iinfo(dtype).max
        image = image.astype(dtype)

        if not scale_factor:
            # scale image to `dtype`'s full range
            scale_factor = int(max_int / np.nanmax(image))
            image = image * scale_factor
        elif isinstance(scale_factor, tuple):
            image = image.astype('float32')
            image = (image - scale_factor[0]) / (scale_factor[1] - scale_factor[0])
            image = np.clip(image, 0, 1) * max_int

        metadata['scale_factor'] = str(scale_factor)

    directory = file.parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    if file.suffix == '.tiff':
        compression: Union[str, None]
        compressionargs: Union[Dict[str, Any], None]
        
        if compress > 0:
            compression = 'zlib'
            compressionargs = {'level': int(compress)}
        else:
            compression = None
            compressionargs = None
        tifffile.imwrite(file.as_posix(), data=image.astype(dtype), compression=compression, compressionargs=compressionargs, metadata=metadata)
    else:
        imwrite(file.as_posix(), image.astype(dtype))


def read_tiff_image(filename: str, dtype: npt.DTypeLike='uint16', scale: bool=True, scale_key: str='scale_factor') -> np.ndarray:
    ''' Load TIFF image data, possibly with scale factor

    Parameters:
    filename (str): path to the image to read
    dtype (npt.DTypeLike): dtype of the output data
    scale (bool): if True, scale image intensity values
    scale_key (str): key of the scaling value in image metadata

    Returns:
    np.ndarray containing image data
    '''

    with tifffile.TiffFile(filename) as tif:
        tmp = tif
        image = tmp.asarray()

    if scale:
        if 'ImageDescription' in tmp.pages[0].tags:
            image_desc_tag = tmp.pages[0].tags['ImageDescription'].value
        elif 'image_description' in tmp.pages[0].tags:
            image_desc_tag = tmp.pages[0].tags['image_description'].as_str()[2:-1]
        image_desc = json.loads(image_desc_tag)

        try:
            scale_factor = int(image_desc[scale_key])
        except ValueError:
            scale_factor = ast.literal_eval(image_desc[scale_key])

        if isinstance(scale_factor, (int, float)):
            image = image / scale_factor
        elif isinstance(scale_factor, tuple):
            iinfo = np.iinfo(image.dtype)
            image = image.astype('float32') / iinfo.max
            image = image * (scale_factor[1] - scale_factor[0]) + scale_factor[0]

    return image.astype(dtype)


def read_image(filename: str, scale_factor: Optional[float]=None, dtype: npt.DTypeLike='uint8') -> np.ndarray:
    ''' Generically read an image using CV2

    Parameters:
    filename (str): path to the image file to read
    scale_factor (float|None): image intensity scaling factor to apply. If None, skip this step
    dtype (npt.DTypeLike): dtype of the output data

    Returns:
    np.ndarray containing image data
    '''
    image = cv2.imread(filename)
    if scale_factor is not None:
        image = (image / scale_factor)

    return image.astype(dtype)
