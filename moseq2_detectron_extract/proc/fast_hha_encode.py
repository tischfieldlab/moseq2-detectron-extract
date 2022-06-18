from typing import Optional
import numpy as np
import cv2

from moseq2_detectron_extract.proc.proc import scale_raw_frames
from moseq2_detectron_extract.proc.roi import plane_ransac


class HHAEncoder():
    def __init__(self, background: np.ndarray, gradient_filter: Optional[int]=7, fix_rotation: bool=False):
        ''' Initialize this encoder

        Parameters:
        background (np.ndarray): background image
        gradient_filter (int): If not None, apply sobel filtering to gradient for normal calculation, should be an odd integer
        fix_rotation (bool): If true, attempt to correct normals relative to axis
        '''
        self.background = background
        self.gradient_filter = gradient_filter
        self.fix_rotation = fix_rotation
        self.R = None

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix


    def compute_normals(self, frame):
        # https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python

        zy, zx = np.gradient(frame)
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        if self.gradient_filter is not None:
            zx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=self.gradient_filter)
            zy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=self.gradient_filter)

        normal = np.dstack((-zx, -zy, np.ones_like(frame)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

        # # offset and rescale values to be in 0-255
        # normal += 1
        # normal /= 2
        # #normal *= 255

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
        acosValue = np.minimum(1, np.maximum(-1, np.sum(tmp, axis=2)))
        angle = np.rad2deg(np.arccos(acosValue.flatten()))
        angle = (angle + 128 - 90)
        angle = angle.reshape(frame.shape)

        return angle

    def compute_height(self, frame):
        h = -frame
        yMin = np.percentile(h[h > 0], 0)
        h = h - yMin
        h[h < 0] = 0
        return h

    def compute_hdisparity(self, frame):
        return self.background - frame

    def encode(self, frame):
        angles = self.compute_normals(frame)
        angles = scale_raw_frames(angles, 0, 180)

        height = self.compute_height(-frame)
        height = scale_raw_frames(height, 0, 100)

        #hdisp = compute_height(-frame)
        hdisp = self.compute_hdisparity(frame)
        hdisp = scale_raw_frames(hdisp, 0, 100)
        #print(angles.shape, angles.dtype, angles.min(), angles.max())
        #print(height.shape, height.dtype, height.min(), height.max())
        #print(hdisp.shape, hdisp.dtype, hdisp.min(), hdisp.max())
        #return np.stack((angles, height, hdisp), axis=2)
        return np.stack((hdisp, height, angles), axis=2)
