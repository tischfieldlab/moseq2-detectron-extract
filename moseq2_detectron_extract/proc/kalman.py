



from typing import Tuple
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from pykalman import KalmanFilter as pyKalmanFilter
from scipy.linalg import block_diag


class KalmanTracker(object):
    ''' Object for kalman tracking
    '''

    def __init__(self, order: int=3, delta_t: float=1.0):
        '''
        Parameters:
        order (int): number between 1 and 4: 1=models position; 2=models position + velocity, 3=models position + velocity + acceleration;
                     4=models position + velocity + acceleration + jerk
        delta_t (float): time between model steps
        '''
        self.order = order
        self.delta_t = delta_t

        # these get set after calling initialize()
        self.kalman_filter: pyKalmanFilter = None
        self.num_points: int = None
        self.init_data: np.ndarray = None

        # these are only used with streaming obervations (i.e. using self.filter_update())
        self.last_mean: np.ndarray = None
        self.last_covar: np.ndarray = None

    @property
    def is_initialized(self) -> bool:
        ''' Has this tracker been initialized?
        '''
        return self.kalman_filter is not None

    def initialize(self, init_data: np.ndarray):
        ''' Initialize the tracker with data
        Parameters:
        init_data (np.ndarray): initial data, of shape (nframes, npoints, 2 [x, y])

        '''
        self.num_points = init_data.shape[1] * 2
        self.init_data = init_data.reshape(init_data.shape[0], -1)

        self.kalman_filter = pyKalmanFilter(
            transition_matrices=self._build_trans_mat(),
            observation_matrices=self._build_observ_matrix(),
            initial_state_mean=self._build_init_state_means(),
        )
        self.kalman_filter.em(self.init_data, n_iter=10)
        state_means, state_covariances = self.kalman_filter.filter(self.init_data)
        self.last_mean = list(state_means)[-1]
        self.last_covar = list(state_covariances)[-1]

    def _build_init_state_means(self) -> np.ndarray:
        ''' Build initial state mean matrix
        '''
        init_state_means = [0] * (self.num_points * self.order)
        for coord_idx, coord_value in enumerate(self.init_data[0]):
            init_state_means[(coord_idx * self.order)] = coord_value
        return np.array(init_state_means)

    def _build_trans_mat(self) -> np.ndarray:
        ''' Build transition matrix
        '''
        derivitives = [
            1.0,                        # position
            self.delta_t,                # velocity
            pow(self.delta_t, 2) / 2,    # acceleration
            pow(self.delta_t, 3) / 6     # jerk
        ]
        trans_mat = []
        for idx in range(0, self.num_points):
            for d in range(self.order):
                t_m = [0] * (self.num_points * self.order)
                depends = range(idx * self.order + d, (idx * self.order) + self.order)
                for i, j in enumerate(depends):
                    t_m[j] = derivitives[i]
                trans_mat.append(t_m)
        return np.array(trans_mat)

    def _build_observ_matrix(self) -> np.ndarray:
        ''' Build observation matrix
        '''
        observ_matrix = []
        for idx in range(0, self.num_points):
            observ_matrix.append([int(x == (idx * self.order)) for x in range(self.num_points * self.order)])
        return np.array(observ_matrix)

    def smooth(self, data: np.ndarray) -> np.ndarray:
        ''' Use the kalman filter to smooth data points
        '''
        to_smooth = data.reshape(data.shape[0], -1)
        means, _ = self.kalman_filter.smooth(to_smooth)
        return means[:, ::self.order].reshape(data.shape)

    def filter(self, data: np.ndarray) -> np.ndarray:
        ''' Filter '''
        to_filter = data.reshape(data.shape[0], -1)
        means, covars = self.kalman_filter.filter(to_filter)
        return means[:, ::self.order].reshape(data.shape)

    def filter_update(self, data: np.ndarray) -> np.ndarray:
        ''' Filter and update '''
        to_filter = data.reshape(-1)
        mean, covar = self.kalman_filter.filter_update(
            self.last_mean,
            self.last_covar,
            to_filter,
        )
        self.last_mean = mean
        self.last_covar = covar
        return mean[::self.order].reshape(data.shape)


class KalmanTracker2(object):
    ''' Object for kalman tracking
    '''

    def __init__(self, order: int=3, delta_t: float=1.0):
        '''
        Parameters:
        order (int): number between 1 and 4: 1=models position; 2=models position + velocity, 3=models position + velocity + acceleration;
                     4=models position + velocity + acceleration + jerk
        delta_t (float): time between model steps
        '''
        self.order = order
        self.delta_t = delta_t

        # these get set after calling initialize()
        self.kalman_filter: KalmanFilter = None
        self.num_points: int = None
        self.init_data: np.ndarray = None

        # these are only used with streaming obervations (i.e. using self.filter_update())
        #self.last_mean: np.ndarray = None
        #self.last_covar: np.ndarray = None

        self.derivitives = [
            1.0,                         # position
            self.delta_t,                # velocity
            pow(self.delta_t, 2) / 2,    # acceleration
            pow(self.delta_t, 3) / 6     # jerk
        ]

    @property
    def is_initialized(self) -> bool:
        ''' Has this tracker been initialized?
        '''
        return self.kalman_filter is not None

    def initialize(self, keypoints: np.ndarray, centroids: np.ndarray, angles: np.ndarray) -> None:
        ''' Initialize the tracker with data
        Parameters:
        keypoints (np.ndarray): keypoint data, of shape (nframes, npoints, 2 [x, y])
        centroids (np.ndarray): centroid data, of shape (nframes, 2 [x, y])
        angles (np.ndarray): angle data, of shape (nframes,)

        '''
        # assert we have the same number of observations for all data sources
        assert keypoints.shape[0] == centroids.shape[0] == angles.shape[0]

        self.init_data = self._prepare_data(keypoints, centroids, angles)
        self.num_points = self.init_data.shape[1]
        self.num_keypoints = keypoints.shape[1]
        assert self.num_points == (keypoints.shape[1] * 2) + 2 + 1

        self.kalman_filter = KalmanFilter(
            dim_x=self.num_points * self.order,
            dim_z=self.num_points,
        )

        # Assign the inital state values
        self.kalman_filter.x = self._build_init_state_means()

        # Assign the state transition matrix
        self.kalman_filter.F = self._build_trans_mat()

        # Assign the measurement function
        self.kalman_filter.H = self._build_observ_matrix()

        # Assign the state covariance matrix
        self.kalman_filter.P = self._build_state_covariance_matrix()

        # Assign the measurement covariance matrix
        self.kalman_filter.R = self._build_measurement_covariance_matrix()

        # Assign process noise
        q = Q_discrete_white_noise(dim=self.order, dt=self.delta_t, var=0.001, block_size=self.num_points)
        self.kalman_filter.Q = q

        # Test validity of matricies
        self.kalman_filter.test_matrix_dimensions(z=self.init_data[0])


    def _build_init_state_means(self) -> np.ndarray:
        ''' Build initial state mean matrix
        '''
        init_state_means = [0] * (self.num_points * self.order)
        for coord_idx, coord_value in enumerate(self.init_data[0]):
            init_state_means[(coord_idx * self.order)] = coord_value
        return np.array(init_state_means)

    def _build_state_covariance_matrix(self, return_obs=False) -> np.ndarray:
        ''' Build state covariance matrix
        '''
        nobs = self.init_data.shape[0]
        obs_data = np.zeros((nobs-1, self.num_points * self.order))
        for coord_idx in range(self.num_points):
            for d in range(self.order):
                col = ((coord_idx * self.order) + d)
                if d == 0:
                    obs_data[:, col] = np.diff(self.init_data[:, coord_idx]) / self.derivitives[d]
                else:
                    obs_data[d:, col] = np.diff(obs_data[d-1:, col - 1]) / self.derivitives[d]
        obs_data = obs_data[self.order:, :] # truncate to only rows we have valid estimates for all derivitives
        #print(obs_data)
        cov = np.cov(obs_data.T, bias=True) * 0.01
        if return_obs:
            return cov, obs_data
        else:
            return cov

    def _build_measurement_covariance_matrix(self) -> np.ndarray:
        ''' Build measurement covariance matrix
        '''
        return np.cov(np.diff(self.init_data.T), bias=True) * 0.1

    def _build_trans_mat(self) -> np.ndarray:
        ''' Build transition matrix
        '''
        trans_mat = []
        for idx in range(0, self.num_points):
            for d in range(self.order):
                t_m = [0] * (self.num_points * self.order)
                depends = range(idx * self.order + d, (idx * self.order) + self.order)
                for i, j in enumerate(depends):
                    t_m[j] = self.derivitives[i]
                trans_mat.append(t_m)
        return np.array(trans_mat)

    def _build_observ_matrix(self) -> np.ndarray:
        ''' Build observation matrix
        '''
        observ_matrix = []
        for idx in range(0, self.num_points):
            observ_matrix.append([int(x == (idx * self.order)) for x in range(self.num_points * self.order)])
        return np.array(observ_matrix)

    def _prepare_data(self, keypoints: np.ndarray, centroids: np.ndarray, angles: np.ndarray) -> np.ndarray:
        if len(keypoints.shape) == 2:
            return np.concatenate((
                keypoints.flatten(),
                centroids.flatten(),
                angles.flatten()
            ))
        else:
            return np.concatenate((
                keypoints.reshape(keypoints.shape[0], -1),
                centroids.reshape(centroids.shape[0], -1),
                angles[:, None]
            ), axis=1)

    def _states_to_measurements(self, states: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.kalman_filter.measurement_of_state, 1, states)

    def _measurements_to_data(self, measures: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Convert a measurements to data
        Parameters:
        measures (np.ndarray): array of shape (nframes, nkeypoints * 2 + 3) OR (nkeypoints * 2 + 3,)
        
        '''
        kpt_idx = range(0, self.num_keypoints * 2)
        cnt_idx = range(kpt_idx[-1] + 1, kpt_idx[-1] + 3)
        ang_idx = range(cnt_idx[-1] + 1, cnt_idx[-1] + 2)
        # print(kpt_idx, cnt_idx, ang_idx)
        # print(measures.shape)
        if len(measures.shape) == 2:
            return (
                measures[:, kpt_idx].reshape(-1, self.num_keypoints, 2), # keypoints
                measures[:, cnt_idx].reshape(-1, 2),                     # centroids
                measures[:, ang_idx].reshape(-1),                        # angles
            )
        else:
            return (
                measures[kpt_idx].reshape(self.num_keypoints, 2), # keypoints
                measures[cnt_idx].reshape(2),                     # centroids
                measures[ang_idx].reshape(1),                     # angles
            )


    def smooth(self, keypoints: np.ndarray, centroids: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Use the kalman filter to smooth data points

        Parameters:
        keypoints (np.ndarray): keypoint data, of shape (nframes, npoints, 2 [x, y])
        centroids (np.ndarray): centroid data, of shape (nframes, 2 [x, y])
        angles (np.ndarray): angle data, of shape (nframes,)
        '''
        to_smooth = self._prepare_data(keypoints, centroids, angles)
        mu, cov, _, _ = self.kalman_filter.batch_filter(to_smooth)
        means, _, _, _ = self.kalman_filter.rts_smoother(mu, cov)
        measurements = self._states_to_measurements(means)
        return self._measurements_to_data(measurements)

    def filter(self, keypoints: np.ndarray, centroids: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Use the kalman filter to batch filter data points
        Parameters:
        keypoints (np.ndarray): keypoint data, of shape (nframes, npoints, 2 [x, y])
        centroids (np.ndarray): centroid data, of shape (nframes, 2 [x, y])
        angles (np.ndarray): angle data, of shape (nframes,)
        '''
        to_filter = self._prepare_data(keypoints, centroids, angles)
        mu, cov, _, _ = self.kalman_filter.batch_filter(to_filter)
        # print(cov)
        measurements = self._states_to_measurements(mu)
        return self._measurements_to_data(measurements)

    def filter_update(self, keypoints: np.ndarray, centroids: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Filter and update '''
        to_filter = self._prepare_data(keypoints, centroids, angles)
        self.kalman_filter.predict()
        self.kalman_filter.update(to_filter)
        measurement: np.ndarray = self.kalman_filter.measurement_of_state(self.kalman_filter.x)
        return self._measurements_to_data(measurement)

    def get_prediction(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu, cov = self.kalman_filter.get_prediction()
        measurement: np.ndarray = self.kalman_filter.measurement_of_state(mu)
        return self._measurements_to_data(measurement)

