from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import numpy.ma as ma
from pykalman import KalmanFilter
from scipy.linalg import block_diag


def timestamps_to_steps(timestamps, step_size=(1/30 * 1000)):
    ''' Generate an array containing the discrete number of time steps between each observation

    Args:
        timestamps: array of observation times
        step_size: the expected delay between time steps

    Returns:
        array with the number of discrete time steps observed between subsequent observations, of shape (timestamps.shape[0] - 1,)
    '''
    return np.rint(np.diff(timestamps) / step_size).astype(int)


def expand_missing_entries(data, time_steps):
    ''' Expand an array `data` to add entries where observations are missing
        as defined by `time_steps`. This is the inverse operation of `reduce_missing_entries()`
        when provided with the same `time_steps`.

        `data` may be n-dimentional, but the first dimention should encode time

        Example:
        > data =    np.array([0, 1, 3, 4, 8, 9, 10])
        > missing = np.array([1, 2, 1, 4, 1, 1])
        > expand_missing_entries(data, missing)
        masked_array(data=[0, 1, --, 3, 4, --, --, --, 8, 9, 10],
                     mask=[False, False,  True, False, False,  True,  True,  True, False, False, False])

        Args:
            data: contiguous array of observations
            time_steps: data defining the number of discrete time steps between observations

        Returns:
            array of shape (sum(time_steps)+1,), with missing observations masked
    '''
    out_shape = (np.sum(time_steps)+1, *data.shape[1:])
    full = np.zeros(out_shape, dtype=data.dtype)
    mask = np.zeros(out_shape, dtype='int')
    i = 0
    j = 0
    for j, k in enumerate(time_steps):
        # print(j, k, i, "->", data[j])
        full[i] = data[j]
        if k == 1:
            mask[i] = 0
        else:
            mask[i+1:i+k] = 1
        i += k
    full[i] = data[j+1]
    return ma.masked_array(full, mask=mask)


def reduce_missing_entries(data, time_steps):
    ''' Reduce an array `data` to remove entries where observations are missing
        as defined by `time_steps`. This is the inverse operation of `expand_missing_entries()`
        when provided with the same `time_steps`.

        `data` may be n-dimentional, but the first dimention should encode time

        Example:
        > data =    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        > missing = np.array([1, 2, 1, 4, 1, 1])
        > reduce_missing_entries(data, missing)
        array([0, 1, 3, 4, 8, 9, 10])

        Args:
            data: contiguous array of observations
            time_steps: data defining the number of discrete time steps between observations

        Returns:
            array of shape (time_steps.shape[0]+1,), with missing observations removed
    '''

    reduced = np.zeros((time_steps.shape[0] + 1, *data.shape[1:]), dtype=data.dtype)
    i = 0
    j = 0
    for j, k in enumerate(time_steps):
        # print(j, k, i, "->", data[i])
        reduced[j] = data[i]
        i += k
    reduced[j+1] = data[i]
    return reduced


def angle_difference(angles1: np.ndarray, angles2: np.ndarray) -> np.ndarray:
    '''Measure the difference between two angles, always returning the smaller difference'''
    diff = (angles2 - angles1) % 360
    to_min = diff > 180
    diff[to_min] = -(360 - diff[to_min])
    return diff


class KalmanTrackerItem(ABC):
    '''Base abstract class for tracking some type of data.
    '''
    def __init__(self, order: int=3, delta_t: float=1.0):
        self.order = order
        self.delta_t = delta_t

    @property
    def state_size(self) -> int:
        '''Tell the size of the state space
        '''
        return self.build_observ_mat().shape[-1]

    @abstractmethod
    def build_trans_mat(self) -> np.ndarray:
        ''' Build transition matrix for this item
        '''
        ...

    @abstractmethod
    def build_observ_mat(self) -> np.ndarray:
        ''' Build observation matrix for this item
        '''
        ...

    @abstractmethod
    def build_init_state_means(self, data: np.ndarray) -> np.ndarray:
        ''' Build the initial state means for this item
        '''
        ...

    def format_data(self, data: np.ndarray) -> np.ndarray:
        '''Format `data` for internal use by the kalman filter.
        '''
        return data

    def inverse_format_data(self, data: np.ndarray) -> np.ndarray:
        '''Format `data` to report only primary state, for returning back to the user.
        '''
        return data[:, ::self.order]


class KalmanTrackerPoint1D(KalmanTrackerItem):
    ''' Kalman tracker item for tracking 1D point
    '''

    def _get_derivitives(self):
        '''Get the derivitives up to `self.order` which define our transition matrix.
        '''
        return [
            1.0,                         # position
            self.delta_t,                # velocity
            pow(self.delta_t, 2) / 2,    # acceleration
            pow(self.delta_t, 3) / 6     # jerk
        ][:self.order]

    def build_trans_mat(self):
        ''' Build transition matrix for this point
        '''
        derivitives = self._get_derivitives()
        trans_mat = np.zeros((self.order, self.order))
        for d in range(self.order):
            depends = range(d, self.order)
            for i, j in enumerate(depends):
                trans_mat[d, j] = derivitives[i]
        return trans_mat

    def build_observ_mat(self):
        ''' Build observation matrix
        '''
        observ_matrix = np.zeros((self.order,))
        observ_matrix[0] = 1
        return observ_matrix

    def build_init_state_means(self, data: np.ndarray):
        init_state_means = np.zeros((self.order,))
        if data.shape[0] > 0:
            init_state_means[0] = data[0]
        else:
            init_state_means[0] = 0
        # Try to estimate initial state means
        #derivitives = self.get_derivitives()
        #for i in range(self.order):
        #    samples = data[:int((i+1 / 2) * 2)+1]
        #    print(samples)
        #    for j in range(i):
        #        samples = np.diff(samples) / derivitives[j]
        #        print(i, j, samples)
        #    init_state_means[i] = samples[0]
        return init_state_means


class KalmanTrackerPoint2D(KalmanTrackerPoint1D):
    ''' Kalman tracker item for tracking a 2D point
    '''

    def build_trans_mat(self):
        ''' Build transition matrix for this point
        '''
        return block_diag(super().build_trans_mat(), super().build_trans_mat())

    def build_observ_mat(self):
        ''' Build observation matrix
        '''
        return block_diag(super().build_observ_mat(), super().build_observ_mat())

    def build_init_state_means(self, data: np.ndarray):
        return np.hstack((super().build_init_state_means(data[:, 0]),
                          super().build_init_state_means(data[:, 1])))


class KalmanTrackerAngle(KalmanTrackerPoint2D):
    ''' Kalman tracker item for tracking an angle
    '''

    def __init__(self, order: int=3, delta_t: float=1.0, degrees: bool=True):
        super().__init__(order=order, delta_t=delta_t)
        self.degrees = degrees

    def build_init_state_means(self, data: np.ndarray):
        ''' Build initial state means
        '''
        return super().build_init_state_means(self.format_data(data))

    def format_data(self, data: np.ndarray):
        if self.degrees:
            data = np.deg2rad(data)
        return np.column_stack([np.sin(data), np.cos(data)])  # convert to y and x coordinates on unit circle

    def inverse_format_data(self, data: np.ndarray):
        data = data[:, ::self.order]  # collect y and x
        data = np.arctan2(data[:, 0], data[:, 1])  # convert to angle
        data = np.where(data < 0 , 2 * np.pi + data, data)  # enforce angle in range [0, 2pi]
        if self.degrees:
            data = np.rad2deg(data)  # convert to degrees
        return data


class KalmanTrackerNPoints2D(KalmanTrackerPoint2D):
    ''' Kalman tracker item for tracking several 2D points
    '''
    def __init__(self, n_points: int, order: int = 3, delta_t: float = 1):
        self.n_points = n_points
        super().__init__(order, delta_t)

    def build_trans_mat(self):
        ''' Build transition matrix for this point
        '''
        trans_mats = []
        for _ in range(self.n_points):
            trans_mats.append(super().build_trans_mat())
        return block_diag(*trans_mats)

    def build_observ_mat(self):
        ''' Build observation matrix
        '''
        observ_mats = []
        for _ in range(self.n_points):
            observ_mats.append(super().build_observ_mat())
        return block_diag(*observ_mats)

    def build_init_state_means(self, data: np.ndarray):
        ''' Build initial state means
        '''
        state_means = []
        for i in range(self.n_points):
            state_means.append(super().build_init_state_means(data[:, i, :]))
        return np.hstack(state_means)

    def format_data(self, data: np.ndarray) -> np.ndarray:
        '''Format `data` for internal use by the kalman filter.
        '''
        return data.reshape(data.shape[0], -1)

    def inverse_format_data(self, data: np.ndarray) -> np.ndarray:
        '''Format `data` to report only primary state, for returning back to the user.
        '''
        return data[:, ::self.order].reshape(data.shape[0], self.n_points, -1)


class KalmanTracker(object):
    ''' Object for kalman tracking
    '''

    def __init__(self, items_to_track: Sequence[KalmanTrackerItem]):
        '''
        Parameters:
        items_to_track (Sequence[KalmanTrackerItem]): Specification of the items to be tracked by this kalman filter
        '''
        if items_to_track is None or len(items_to_track) <= 0:
            raise ValueError("You need to supply a list of `KalmanTrackerItem`s to the constructor!")

        timesteps = [item.delta_t for item in items_to_track]
        if not np.allclose(timesteps, timesteps[0]):
            raise ValueError(f"Timesteps across `KalmanTrackerItem` must be the same! Got: {', '.join([str(t) for t in timesteps])}")

        self.items = items_to_track

        # these get set after calling initialize()
        self.kalman_filter: KalmanFilter = None

        # these are only used with streaming obervations (i.e. using self.filter_update())
        self.last_mean: np.ndarray
        self.last_covar: np.ndarray

    @property
    def is_initialized(self) -> bool:
        ''' Has this tracker been initialized?
        '''
        return self.kalman_filter is not None

    def initialize(self, init_data: Sequence[np.ndarray]) -> None:
        ''' Initialize the tracker with data
        Parameters:
        init_data (Sequence[np.ndarray]): initial data, one element in the sequence for each `KalmanTrackerItem`
        '''
        # check the init data shape looks right
        if len(init_data) != len(self.items):
            raise ValueError(f"Length of `init_data` ({len(init_data)}) does not equal length of `items_to_track` ({len(self.items)})")

        # init the kalman filter
        self.kalman_filter = KalmanFilter(
            transition_matrices=self._build_trans_mat(),
            observation_matrices=self._build_observ_matrix(),
            initial_state_mean=self._build_init_state_means(init_data),
            em_vars=['transition_covariance', 'observation_covariance', 'initial_state_covariance']
        )

        # filter out any non-finite values and run em (if there are any values left)
        formatted_init_data = self._format_data(init_data)
        finite_values = np.isfinite(formatted_init_data).any(axis=1)
        if np.count_nonzero(finite_values) > 0:
            self.kalman_filter.em(formatted_init_data[finite_values], n_iter=10)

        # set the last mean and covar to the initial states, for use by *_update methods
        self.last_mean = self.kalman_filter.initial_state_mean
        self.last_covar = self.kalman_filter.initial_state_covariance

    def _build_init_state_means(self, init_data: Sequence[np.ndarray]) -> np.ndarray:
        ''' Build initial state mean matrix
        '''
        return np.hstack([itm.build_init_state_means(init_data[i]) for i, itm in enumerate(self.items)])

    def _build_trans_mat(self) -> np.ndarray:
        ''' Build transition matrix
        '''
        return block_diag(*[itm.build_trans_mat() for itm in self.items])

    def _build_observ_matrix(self) -> np.ndarray:
        ''' Build observation matrix
        '''
        return block_diag(*[itm.build_observ_mat() for itm in self.items])

    def _format_data(self, data: Sequence[np.ndarray]) -> np.ndarray:
        fmt = np.column_stack([itm.format_data(data[i]) for i, itm in enumerate(self.items)])
        fmt = ma.masked_invalid(fmt)
        return fmt

    def _inverse_format_data(self, data: np.ndarray) -> Sequence[np.ndarray]:
        out = []
        offset = 0
        for itm in self.items:
            out.append(itm.inverse_format_data(data[:, offset:offset+itm.state_size]))
            offset += itm.state_size
        return out

    def sample(self, n_timesteps: int = 1, init_data: np.ndarray = None) -> Sequence[np.ndarray]:
        '''Use the kalman filter to look into the future'''
        if init_data is not None:
            init_state = self._build_init_state_means(init_data)
        else:
            init_state = self.last_mean
        states, _ = self.kalman_filter.sample(n_timesteps, init_state)
        return self._inverse_format_data(states)

    def smooth(self, data: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        ''' Use the kalman filter to smooth data points
        '''
        to_smooth = self._format_data(data)
        means, _ = self.kalman_filter.smooth(to_smooth)
        return self._inverse_format_data(means)

    def smooth_update(self, data: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        '''Smooth data and update the current state
        '''
        to_smooth = self._format_data(data)
        means, covars = self.kalman_filter.smooth(to_smooth)
        self.last_mean = self.kalman_filter.initial_state_mean = means[-1]
        self.last_covar = self.kalman_filter.initial_state_covariance = covars[-1]
        return self._inverse_format_data(means)

    def filter(self, data: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        ''' Filter '''
        to_filter = self._format_data(data)
        means, _ = self.kalman_filter.filter(to_filter)
        return self._inverse_format_data(means)

    def filter_update(self, data: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        ''' Filter and update '''
        to_filter = self._format_data(data)[0]
        mean, covar = self.kalman_filter.filter_update(
            self.last_mean,
            self.last_covar,
            to_filter,
        )
        self.last_mean = mean
        self.last_covar = covar
        return self._inverse_format_data(mean[None, :])

    def __str__(self) -> str:
        buffer = ""
        buffer += "transition_matrices:\n"
        buffer += str(self.kalman_filter.transition_matrices)
        buffer += "\n\n"

        buffer += "observation_matrices:\n"
        buffer += str(self.kalman_filter.observation_matrices)
        buffer += "\n\n"

        buffer += "transition_covariance:\n"
        buffer += str(self.kalman_filter.transition_covariance)
        buffer += "\n\n"

        buffer += "observation_covariance:\n"
        buffer += str(self.kalman_filter.observation_covariance)
        buffer += "\n\n"

        buffer += "transition_offsets:\n"
        buffer += str(self.kalman_filter.transition_offsets)
        buffer += "\n\n"

        buffer += "observation_offsets:\n"
        buffer += str(self.kalman_filter.observation_offsets)
        buffer += "\n\n"

        buffer += "initial_state_mean:\n"
        buffer += str(self.kalman_filter.initial_state_mean)
        buffer += "\n\n"

        buffer += "initial_state_covariance:\n"
        buffer += str(self.kalman_filter.initial_state_covariance)
        buffer += "\n\n"

        buffer += "n_dim_state:\n"
        buffer += str(self.kalman_filter.n_dim_state)
        buffer += "\n\n"

        buffer += "n_dim_obs:\n"
        buffer += str(self.kalman_filter.n_dim_obs)
        buffer += "\n\n"
        return buffer

    def __repr__(self) -> str:
        return str(self)
