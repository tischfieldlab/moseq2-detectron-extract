import enum
from typing import Optional
import numpy as np
import cv2

from moseq2_detectron_extract.proc.proc import scale_raw_frames
from moseq2_detectron_extract.proc.roi import plane_ransac


class HHAEncoder(object):
    ''' Class providing HHA encoding of depth images
    '''
    def __init__(self, background: np.ndarray, gradient_filter: Optional[int]=None, gradient_step: Optional[int]=3, fix_rotation: bool=False):
        ''' Initialize this encoder

        Parameters:
        background (np.ndarray): background image
        gradient_filter (int): If not None, apply sobel filtering to gradient for normal calculation, should be an odd integer
        fix_rotation (bool): If true, attempt to correct normals relative to axis
        '''
        self.background = background
        self.gradient_filter = gradient_filter
        self.gradient_step = gradient_step
        self.fix_rotation = fix_rotation
        self.R = None

    def rotation_matrix_from_vectors(self, vec1, vec2):
        ''' Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        '''
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix


    def compute_normals(self, frame: np.ndarray) -> np.ndarray:
        ''' compute normals from a depth frame

        Parameters:
        frame (np.ndarray): depth frame to compute normals upon, of shape (height, width)

        Returns:
        np.ndarray: angles of the computed gradiant
        '''
        # https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python

        zy, zx = np.gradient(frame, self.gradient_step)
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        if self.gradient_filter is not None:
            zx = cv2.Sobel(zx, cv2.CV_64F, 1, 0, ksize=self.gradient_filter)
            zy = cv2.Sobel(zy, cv2.CV_64F, 0, 1, ksize=self.gradient_filter)

        normal = np.dstack((-zx, -zy, np.ones_like(frame)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

        if self.fix_rotation:
            # align to estimated level
            if self.R is None:
                plane, _ = plane_ransac(frame.astype(float), depth_range=(0, 300), mask=(frame>0))
                self.R = self.rotation_matrix_from_vectors(plane[:3], [1, 0, 0])

            normal = normal.reshape((-1, 3))
            normal = np.apply_along_axis(self.R.dot, axis=1, arr=normal)
            normal = normal.reshape((*frame.shape, 3))

        # estimate angle relative to some reference
        tmp = np.multiply(normal, np.array([0, 1, 0]))
        acos_value = np.minimum(1, np.maximum(-1, np.sum(tmp, axis=2)))
        angle = np.rad2deg(np.arccos(acos_value.flatten()))
        angle = (angle + 128 - 90)
        angle = angle.reshape(frame.shape)
        angle = angle.clip(*np.percentile(angle, [5, 95]))

        # estimate magnitude to some reference
        magnitude = np.multiply(normal, np.array([0, 0, 1])).sum(axis=2)
        magnitude = magnitude.clip(np.percentile(magnitude, 5, None))

        return angle, magnitude

    def compute_height(self, frame: np.ndarray) -> np.ndarray:
        ''' Compute height from depth frame
        '''
        height = -frame
        y_min = np.percentile(height[height > 0], 1)
        height = height - y_min
        height[height < 0] = 0
        return height

    def compute_hdisparity(self, frame: np.ndarray) -> np.ndarray:
        ''' Compute horizontal disparity from depth frame
        '''
        return self.background - frame

    def encode(self, frame: np.ndarray, mode='mda') -> np.ndarray:
        ''' Encode a depth frame
        '''
        if 'm' in mode or 'a' in mode:
            angles, magnitudes = self.compute_normals(frame)

        data = []
        for m in mode:
            if m == 'm': # magnitudes
                magnitudes = scale_raw_frames(magnitudes, 0, 1)
                data.append(magnitudes)

            elif m == 'a': # angles
                angles = scale_raw_frames(angles, 0, 180)
                data.append(angles)

            elif m == 'd': # depth
                depth = self.compute_height(-frame)
                depth = scale_raw_frames(depth, 0, 100)
                data.append(depth)

            elif m == 'h': #height
                height = self.compute_hdisparity(frame)
                height = scale_raw_frames(height, 0, 100)
                data.append(height)

        return np.stack(data, axis=2)
