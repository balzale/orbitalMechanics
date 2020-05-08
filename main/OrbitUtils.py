import numpy as np
import scipy as sp


def state_vector_derivative(x0, t0, mu):
    """Returns the derivative of the position and velocity of the satellite.

    x0 = state vector as [rx ry rz vx vy vz]
    mu = Standard gravitational parameter
    """
    x, xdot = x0[:3], x0[3:]
    xdotdot = -mu / (np.linalg.norm(x)) ** 3 * x
    return np.r_[xdot, xdotdot]


def wrap_to_180(angle):
    """This function wrap the input angle between -180° and 180.

    INPUT:
    -------
    angle [deg]: float or array_like
        angle (or array of angles) that needs to be wrapped.

    OUTPUT:
    -------
    wrapped_angle [deg]: float or array_like
         wrapped angle (or array of angle) between -180° and 180°.
    """
    wrapped_angle = (angle + 180) % 360 - 180
    return wrapped_angle


def state_vector2orbital_elements(radius_vect, velocity_vect, mu):
    """Returns the orbital elements given the state vector.

    INPUT:
    ------
    radius_vect	= position vector [rx, ry, rz]		- [km]
    velocity_vect	= velocity vector [vx, vy, vz]		- [km/s]
    mu		= Standard gravitational parameter	- [km^3/s^2]

    OUTPUT:
    -------
    orbital_elements	= [inclination [rad],
              angular_momentum_norm [km^2/s],
              right_ascension_ascending_node [rad],
              eccentricity_norm,
              argument_of_perigee [rad],
              true_anomaly [rad]]
    """
    radius_norm = np.linalg.norm(radius_vect)
    velocity_norm = np.linalg.norm(velocity_vect)
    radial_velocity = np.dot(radius_vect, velocity_vect) / radius_norm
    angular_momentum_vect = np.cross(radius_vect, velocity_vect)
    angular_momentum_norm = np.linalg.norm(angular_momentum_vect)
    inclination = np.arccos(angular_momentum_vect[2] / angular_momentum_norm)
    z_axis = np.array([0, 0, 1])
    node_line = np.cross(z_axis, angular_momentum_vect)
    node_line_norm = np.linalg.norm(node_line)
    if node_line[1] >= 0:
        right_ascension = np.arccos(node_line[0] / node_line_norm)
    else:
        right_ascension = 2 * np.pi - np.arccos(node_line[0] / node_line_norm)
    eccentricity_vector = 1 / mu * (
            (velocity_norm ** 2 - mu / radius_norm) * radius_vect - radius_norm * radial_velocity * velocity_vect)
    eccentricity_norm = np.linalg.norm(eccentricity_vector)
    if eccentricity_vector[2] >= 0:
        argument_of_perigee = np.arccos(np.dot(node_line, eccentricity_vector) / (node_line_norm * eccentricity_norm))
    else:
        argument_of_perigee = 2 * np.pi - np.arccos(
            np.dot(node_line, eccentricity_vector) / (node_line_norm * eccentricity_norm))
    if radial_velocity >= 0:
        true_anomaly = np.arccos(np.dot(eccentricity_vector, radius_vect) / (radius_norm * eccentricity_norm))
    else:
        true_anomaly = 2 * np.pi - np.arccos(np.dot(eccentricity_vector, radius_vect) /
                                             (radius_norm * eccentricity_norm))
    orbital_elements = np.array(
        [inclination, angular_momentum_norm, right_ascension, eccentricity_norm, argument_of_perigee, true_anomaly])
    return orbital_elements


def orbital_elements2state_vector(inclination, angular_momentum_norm, right_ascension, eccentricity_norm,
                                  argument_of_perigee, true_anomaly, mu):
    """Returns the state vector given the six orbital elements.

    INPUT:
    ------
    inclination		= Inclination [rad],
    angular_momentum_norm	= Angular momentum norm [km^2/s],
    right_ascension_ascending_node 		= Right ascension [rad],
    eccentricity_norm 	= Eccentricity norm,
    argument_of_perigee 	= Argument of the perigee [rad],
    true_anomaly 		= True anomaly [rad]
    mu 			= Standard gravitational parameter [km^3/s^2]

    OUTPUT:
    -------
    state_vector		= State vector [rx, ry, rz, vx, vy, vz]	- [km, km/s]
    """
    radius_vector_perifocal = angular_momentum_norm ** 2 / (mu * (1 + eccentricity_norm * np.cos(true_anomaly))) * \
        np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0])
    velocity_vector_perifocal = mu / angular_momentum_norm * np.array(
        [-np.sin(true_anomaly), eccentricity_norm + np.cos(true_anomaly), 0])
    rot_matrix_perifocal2_geocentric_equat = np.array([[np.cos(right_ascension) * np.cos(argument_of_perigee) -
                                                        np.sin(right_ascension) * np.sin(argument_of_perigee) *
                                                        np.cos(inclination), -np.cos(right_ascension) *
                                                        np.sin(argument_of_perigee) - np.sin(right_ascension) *
                                                        np.cos(inclination) * np.cos(argument_of_perigee),
                                                        np.sin(right_ascension) * np.sin(inclination)],
                                                       [np.sin(right_ascension) * np.cos(argument_of_perigee) +
                                                        np.cos(right_ascension) * np.cos(inclination) *
                                                        np.sin(argument_of_perigee), -np.sin(right_ascension) *
                                                        np.sin(argument_of_perigee) + np.cos(right_ascension) *
                                                        np.cos(inclination) * np.cos(argument_of_perigee),
                                                        -np.cos(right_ascension) * np.sin(inclination)],
                                                       [np.sin(inclination) * np.sin(argument_of_perigee),
                                                        np.sin(inclination) * np.cos(argument_of_perigee),
                                                        np.cos(inclination)]])
    radius_vector_geocentric_equat = rot_matrix_perifocal2_geocentric_equat.dot(radius_vector_perifocal)
    velocity_vector_geocentric_equat = rot_matrix_perifocal2_geocentric_equat.dot(velocity_vector_perifocal)
    state_vector = np.array([radius_vector_geocentric_equat, velocity_vector_geocentric_equat])
    return state_vector


def object_sky_locator(object_latitude, object_longitude, observer_latitude, observer_longitude):
    """This function returns the azimuth and the elevation over the horizon of an object in the sky,
    given its latitude and longitude and the coordinates of the observer.

    INPUT:
    ------
    object_latitude     = latitude of the object to be observed in the sky.     [deg]
    object_longitude    = longitude of the object to be observed in the sky.    [deg]
    observer_latitude   = latitude of the observer on the earth surface.        [deg]
    observer_longitude  = longitude of the observer on the earth surface.       [deg]

    OUTPUT:
    ------
    azimuth_deg         = azimuth of the observed object. Range: 0°->360°               [deg]
    elevation_deg       = elevation over the horizon of the object. Range: -90°->+90°   [deg]
    """
    object_latitude_rad = np.deg2rad(object_latitude)
    object_longitude_rad = np.deg2rad(object_longitude)
    observer_latitude_rad = np.deg2rad(observer_latitude)
    observer_longitude_rad = np.deg2rad(observer_longitude)
    delta_longitude_obj_obs_rad = np.abs(object_longitude_rad - observer_longitude_rad)
    object_observer_angle_on_geodetic = np.arccos(
        np.sin(object_latitude_rad) * np.sin(observer_latitude_rad) +
        np.cos(object_latitude_rad) * np.cos(observer_latitude_rad) *
        np.cos(delta_longitude_obj_obs_rad))
    elevation_rad = np.arcsin(np.sin(object_latitude_rad) * np.sin(observer_latitude_rad) +
                              np.cos(object_latitude_rad) * np.cos(observer_latitude_rad) *
                              np.cos(delta_longitude_obj_obs_rad))
    sin_explementary_of_azimuth = np.sin(delta_longitude_obj_obs_rad) * np.cos(
        object_latitude_rad) / np.sin(elevation_rad)
    cos_explementary_of_azimuth = \
        np.sin(object_latitude_rad) - np.sin(
            observer_latitude_rad) * np.cos(
            object_observer_angle_on_geodetic) / \
        (np.cos(observer_latitude_rad) * np.sin(object_observer_angle_on_geodetic))
    azimuth_rad = 2 * np.pi - np.arctan(sin_explementary_of_azimuth, cos_explementary_of_azimuth)
    elevation_deg = np.rad2deg(elevation_rad)
    azimuth_deg = np.rad2deg(azimuth_rad)
    return azimuth_deg, elevation_deg


class MeanAnomalyTrueAnomaly:
    """
    This class can be used to compute the true anomaly given the mean anomaly and vice versa. The only input needed is
    the eccentricity norm of the orbit. Both are given in radians.

    Attributes
    ----------
    meanAnomaly : float
        mean anomaly of the orbit [rad]
    trueAnomaly : float
        true anomaly of the orbit [rad]
    eccentricityNorm : float
        norm of the eccentricity of the orbit [-]

    Methods
    -------
    set_mean_anomaly(mean_anomaly)
        Set the mean anomaly and change accordingly the true anomaly attribute.
    set_true_anomaly(true_anomaly)
        Set the true anomaly and change accordingly the mean anomaly attribute.
    """

    def __init__(self, eccentricity_norm):
        """
        Parameters
        ----------
        eccentricity_norm : float
            Norm of the eccentricity of the orbit.
        """
        self.meanAnomaly = 0.0
        self.trueAnomaly = 0.0
        self.eccentricityNorm = eccentricity_norm

    def set_mean_anomaly(self, mean_anomaly):
        """This function sets the mean anomaly and compute and change accordingly the true anomaly.

        INPUT:
        ----------
        mean_anomaly [rad]: float
        """
        self.meanAnomaly = mean_anomaly
        if self.meanAnomaly >= np.pi:
            eccentric_anomaly_zero = self.meanAnomaly - self.eccentricityNorm / 2
        else:
            eccentric_anomaly_zero = self.meanAnomaly + self.eccentricityNorm / 2
        eccentric_anomaly = sp.optimize.newton(self._mean_anomaly_func, eccentric_anomaly_zero,
                                               self._mean_anomaly_derivative, tol=1e-6)
        self.trueAnomaly = 2 * np.arctan(np.sqrt((1 + self.eccentricityNorm) / (1 - self.eccentricityNorm))
                                         * np.tan(eccentric_anomaly / 2))

    def set_true_anomaly(self, true_anomaly):
        """This function sets the true anomaly and compute and change accordingly the mean anomaly.

        INPUT:
        ----------
        true_anomaly [rad]: float
        """
        self.trueAnomaly = true_anomaly
        eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - self.eccentricityNorm) / (1 + self.eccentricityNorm)) *
                                          np.tan(self.trueAnomaly / 2))
        self.meanAnomaly = eccentric_anomaly - self.eccentricityNorm * np.sin(eccentric_anomaly)

    def _mean_anomaly_func(self, eccentric_anomaly):
        mean_anomaly = eccentric_anomaly - self.eccentricityNorm * np.sin(eccentric_anomaly) - self.meanAnomaly
        return mean_anomaly

    def _mean_anomaly_derivative(self, eccentric_anomaly):
        mean_anomaly_derivative = 1 - self.eccentricityNorm * np.cos(eccentric_anomaly)
        return mean_anomaly_derivative
