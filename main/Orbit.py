import numpy as np
import OrbitUtils as orbut
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import scipy.integrate as spi
import datetime as dt
import julianDay as jd
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# TODO: create a different class from Orbit around Earth and other planets

class Orbit:
    """This Class allows to create an orbit around an object, specifying the six orbital elements and the epoch
    """

    def __init__(self, planet, inclination, angular_momentum_norm, right_ascension_ascending_node,
                 eccentricity_norm, argument_of_perigee, true_anomaly, epoch=dt.datetime.now()):
        # TODO: add full description of the class and its functions. epoch is datetime type.
        self.planet = planet
        self.epoch = epoch
        self.inclination = inclination
        self.angularMomentumNorm = angular_momentum_norm
        self.rightAscension = right_ascension_ascending_node
        self.eccentricityNorm = eccentricity_norm
        self.argumentOfPerigee = argument_of_perigee
        self.trueAnomaly = true_anomaly
        self.position = self._get_state_vector()[0]
        self.velocity = self._get_state_vector()[1]
        self.meanAnomaly = self._get_mean_anomaly()
        self.orbitPeriod = self._get_period()

    def plot_orbit_perifocal(self):
        """This function plots the orbit in the perifocal frame.

        !!! IMPORTANT !!!:  Add at the beginning of the script:     include import matplotlib.pyplot as plt
                            Add at the end of the script:           plt.show()
        """
        plt.figure()
        theta_vector = np.linspace(self.trueAnomaly, self.trueAnomaly + 2 * np.pi, 1000)
        position_x = self.angularMomentumNorm ** 2 / self.planet.gravitationalParameter * np.cos(theta_vector) / \
            (1 + self.eccentricityNorm * np.cos(theta_vector))
        position_y = self.angularMomentumNorm ** 2 / self.planet.gravitationalParameter * np.sin(theta_vector) / \
            (1 + self.eccentricityNorm * np.cos(theta_vector))
        # Circle representing the planet and plot of the orbit.
        planet_circle = plt.Circle((0, 0), radius=self.planet.radius, color='black', alpha=0.7)
        plt.gca().add_patch(planet_circle)
        plt.plot(position_x, position_y, color='#1f77b4', linewidth=2)
        plt.plot(position_x[0], position_y[0], 'h', color='#1e1596', markersize=8)
        plt.grid()
        plt.axis("equal")

    def plot_orbit_3d(self):
        """This function plot the orbit in 3D.

        !!! IMPORTANT !!!:  Add at the beginning of the script:     include import matplotlib.pyplot as plt
                            Add at the end of the script:           plt.show()
        """
        r_plot = self._get_position_vector_one_orbit()
        plt.figure()
        ax = plt.axes(projection='3d')

        # Creation and plot of the planet sphere.
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        xm = self.planet.radius * np.outer(np.cos(phi), np.sin(theta))
        ym = self.planet.radius * np.outer(np.sin(phi), np.sin(theta))
        zm = self.planet.radius * np.outer(np.ones(np.size(phi)), np.cos(theta))
        ax.plot_surface(xm, ym, zm)

        # Plot of the orbit.
        ax.plot3D(r_plot[0][:], r_plot[1][:], r_plot[2][:], 'r')
        plt.grid()

    def set_new_epoch(self, new_epoch=dt.datetime.now()):
        """This function sets a new desired epoch and updates the mean & true anomaly and the state vector of
        the satellite. If not specified, it sets the epoch to the current date and time.

        INPUT:
        ------
        new_epoch: datetime
            new desired epoch. (Default = dt.datetime.now())
        """
        new_mean_anomaly = self.meanAnomaly + 2 * np.pi / self.orbitPeriod * ((new_epoch - self.epoch).total_seconds())
        self.epoch = new_epoch
        self._set_mean_anomaly(new_mean_anomaly)

    def satellite_ground_track(self, epoch_ground_track=dt.datetime.now(), plot_map=True):
        """This function returns the satellite latitude and longitude during one orbit at the specified epoch.
        If the epoch is not specified, current time is consider.

        INPUT:
        ------
        epoch_ground_track: datetime
            epoch at which the ground track is plotted. Default: dt.datetime.now()
        plot_map: bool
            if True, two map projections are plotted with the satellite ground track and its footprint. (Default: True)

        OUTPUT:
        -----
        latitude_satellite [deg]: array_like
           vector of latitude of the satellite during one orbit.
        longitude_satellite [deg]: array_like
            vector of longitude of the satellite during one orbit.

        !!! IMPORTANT !!!:  Add at the beginning of the script:     include import matplotlib.pyplot as plt
                            Add at the end of the script:           plt.show()
        """
        # TODO: add the minimum inclination needed to see the satellite
        # The current value of epoch is saved in a temporary variable and all the Orbit attributes are update
        # accordingly with the required epoch_ground_track, epoch at which the ground track is plotted.
        old_epoch = self.epoch
        self.set_new_epoch(epoch_ground_track)

        # Now that a mean anomaly at epoch has been set, the right ascension
        # (between 0-360° with the modulus operator %) and the latitude_satellite  are computed along the full orbit.
        latitude_satellite = self._get_declination_one_orbit()
        right_ascension = self._get_right_ascension_one_orbit() % 360

        # The greenwich sideral time is computed for each time step which the orbit time has been divided in.
        greenwich_sideral_time_vector = self._get_greenwich_sideral_time(epoch_ground_track)

        # Longitude of the Satellite, wrapped between -180° and 180°.
        longitude_satellite = orbut.wrap_to_180(right_ascension - greenwich_sideral_time_vector)

        # Computation of the satellite footprint.
        latitude_footprint, longitude_footprint = self._get_footprint_satellite(latitude_satellite[0],
                                                                                longitude_satellite[0])
        if plot_map:
            # Plot the figure of the track with the map on the Orthographic and PlateCarree.
            self._plot_orthographic_projection(latitude_satellite, longitude_satellite,
                                               latitude_footprint, longitude_footprint, epoch_ground_track)
            self._plot_plate_carree_projection(latitude_satellite, longitude_satellite,
                                               latitude_footprint, longitude_footprint, epoch_ground_track)

        # The epoch is set back to its actual value.
        self.set_new_epoch(old_epoch)
        return latitude_satellite, longitude_satellite

    def get_right_ascension(self):
        """This function returns the right ascension of the satellite.

        OUTPUT:
        ------
        alpha [deg]: float
            right ascension of the satellite in degrees.
        """
        r = self.position
        alpha = np.rad2deg(np.arctan2(r[1], r[0]))
        return alpha

    def get_declination(self):
        """This function returns the declination of the satellite.

        OUTPUT:
        ------
        delta [deg]: float
            declination of the satellite [deg]
        """
        r = self.position
        radius_norm = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        delta = np.rad2deg(np.arcsin(r[2] / radius_norm))
        return delta

    def is_satellite_in_light(self):
        """Description here. Remind if self.planet is Sun, the algorith from Curtis"""
        if self.planet.name == "Sun":
            in_light = True
        else:
            position_vect_satellite = self.position
            position_vect_sun = -self.planet.position
            # Angle between the satellite position vector and the Sun-planet position vector.
            theta = np.arccos(np.dot(position_vect_satellite, position_vect_sun) /
                              (np.linalg.norm(position_vect_satellite) * np.linalg.norm(position_vect_sun)))
            # Angle between Sun-planet position and radius of the planet
            theta_one = np.arccos(self.planet.radius/np.linalg.norm(position_vect_sun))
            # Angle between Satellite position and radius of the planet
            theta_two = np.arccos(self.planet.radius/np.linalg.norm(position_vect_satellite))
            theta_sum = theta_one + theta_two
            if theta_sum <= theta:
                in_light = False
            else:
                in_light = True
        return in_light

    def is_satellite_visible(self, station_latitude, station_longitude, epoch_ground_station=dt.datetime.now(),
                             plot_map=True):
        """This method verifies if a satellite is visible from a ground station at epoch.

        INPUT:
        ------
        observer_latitude [deg]: float
            latitude of the ground station in degrees. Positive for North, negative for South.
        observer_longitude [deg]: float
            longitude of the ground station in degrees. Between 180° and -180°.
        epoch_ground_station: datetime
            epoch of interest.
        plot_map: bool
            if True, two map projections are plotted showing the satellite ground track,
            its footprint and the ground station position. (Default: True)

        OUTPUT:
        ------
        station_visible: bool
            True if  the satellite is visible from the ground station at epoch.
            False if the satellite is NOT visible from the ground station at epoch.

        !!! IMPORTANT !!!:  Add at the beginning of the script:     include import matplotlib.pyplot as plt
                             Add at the end of the script:           plt.show()
        """
        # The ground track is first computed and plotted.
        latitude_sat, longitude_sat = self.satellite_ground_track(epoch_ground_track=epoch_ground_station,
                                                                  plot_map=plot_map)
        if plot_map:
            # Number Id of the figures opened by the method self.plot_ground_track is retrieved.
            plate_carree_figure_number = plt.gcf().number
            orthographic_figure_number = plate_carree_figure_number - 1
        # A dummy mean anomaly is computed and set in the Orbit.meanAnomaly. The real value is saved and re-set back
        # into the Orbit class at the end of the method.
        real_mean_anomaly = self.meanAnomaly
        dummy_mean_anomaly = self.meanAnomaly + 2 * np.pi / self.orbitPeriod * ((epoch_ground_station - self.epoch)
                                                                                .total_seconds())
        self._set_mean_anomaly(dummy_mean_anomaly)

        satellite_current_declination = self.get_declination()
        satellite_current_right_ascension = self.get_right_ascension() % 360

        # The greenwich sideral time at epoch is computed.
        greenwich_sideral_time = self._get_greenwich_sideral_time(epoch_ground_station)[0]

        # Longitude of the Satellite.
        satellite_current_longitude = satellite_current_right_ascension - greenwich_sideral_time

        # Longitude between -180 and 180.
        satellite_current_longitude = orbut.wrap_to_180(satellite_current_longitude)

        delta_longitude_satellite_station = np.abs(satellite_current_longitude - station_longitude)

        # Conversion of some variables from deg to rad.
        satellite_current_declination_rad = np.deg2rad(satellite_current_declination)
        station_latitude_rad = np.deg2rad(station_latitude)
        delta_longitude_sat_station_rad = np.deg2rad(delta_longitude_satellite_station)

        radius_satellite = self.position
        max_central_angle = np.arccos(self.planet.radius / np.linalg.norm(radius_satellite))
        station_satellite_angle_on_geodetic = np.arccos(
            np.sin(satellite_current_declination_rad) * np.sin(station_latitude_rad) +
            np.cos(satellite_current_declination_rad) * np.cos(station_latitude_rad) *
            np.cos(delta_longitude_sat_station_rad))

        azimuth, elevation = orbut.object_sky_locator(latitude_sat, longitude_sat, station_latitude, station_latitude)
        # TODO: the object_sky_locator function must be tested against known data.

        if station_satellite_angle_on_geodetic < max_central_angle:
            station_visible = True
            # TODO: the figure of the satellite elevation/azimuth must be plotted.
        else:
            station_visible = False

        if plot_map:
            if station_visible:
                plt.figure(plate_carree_figure_number)
                plt.plot(station_longitude, station_latitude, '*', color='g', markersize=10, markeredgewidth=2)
                plt.figure(orthographic_figure_number)
                station_points = ccrs.Orthographic(longitude_sat[0], latitude_sat[0]).transform_point(station_longitude,
                                                                                                      station_latitude,
                                                                                                      ccrs.Geodetic())
                plt.plot(station_points[0], station_points[1], '*', color='g', markersize=10, markeredgewidth=2)
            else:
                plt.plot(station_longitude, station_latitude, 'x', color='r', markersize=8, markeredgewidth=2)
                plt.figure(orthographic_figure_number)
                station_points = ccrs.Orthographic(longitude_sat[0], latitude_sat[0]).transform_point(station_longitude,
                                                                                                      station_latitude,
                                                                                                      ccrs.Geodetic())
                plt.plot(station_points[0], station_points[1], 'x', color='r', markersize=8, markeredgewidth=2)

        # The mean anomaly is set back to its actual value.
        self._set_mean_anomaly(real_mean_anomaly)
        return station_visible

    def _get_state_vector(self):
        """Returns the state vector of the satellite.

            OUTPUT:
            -------
            state_vector [km, km/s]: array_like
                State vector [[rx, ry, rz], [vx, vy, vz]]	- [[km], [km/s]]
            """
        state_vector = orbut.orbital_elements2state_vector(self.inclination, self.angularMomentumNorm,
                                                           self.rightAscension, self.eccentricityNorm,
                                                           self.argumentOfPerigee, self.trueAnomaly,
                                                           self.planet.gravitationalParameter)
        return state_vector

    def _set_mean_anomaly(self, mean_anomaly):
        """This function set the mean anomaly and update the new true anomaly, the position vector and
        the velocity vector.

        INPUT:
        ------
        meanAnomaly [rad]: float
            mean anomaly
        """
        self.meanAnomaly = mean_anomaly
        # Creating a support Class to compute the true anomaly from the mean anomaly and eccentricity.
        mean_and_true = orbut.MeanAnomalyTrueAnomaly(self.eccentricityNorm)
        mean_and_true.set_mean_anomaly(mean_anomaly)
        self.trueAnomaly = mean_and_true.trueAnomaly
        self._set_position_vector()
        self._set_velocity_vector()

    def _set_position_vector(self):
        """This functions simply set the new satellite position vector, according to its orbital elements
        """
        self.position = self._get_state_vector()[0]

    def _set_velocity_vector(self):
        """This functions simply set the new satellite velocity vector, according to its orbital elements
        """
        self.velocity = self._get_state_vector()[1]

    def _get_footprint_satellite(self, latitude_sat, longitude_sat):
        """This function returns the two vectors of latitude and associated longitude for the boundary of the satellite
        footprint, given the current latitude and longitude of the satellite.

        INPUT:
            ------
            latitude_sat [deg]: float
                Current latitude of the satellite.
            longitude_sat [deg]: float
                Current longitude of the satellite.

            OUTPUT:
            -------
            latitude_footprint [deg]: array_like
                vector of latitudes of the boundary of the satellite footprint.
            longitude_footprint	[deg]
                vector of longitudes of the boundary of the satellite footprint.
        """
        radius_satellite = self.position
        # delta_latitude is the latitude, which added to the current satellite latitude,
        # gives the max latitude of visibility.
        delta_latitude = np.arccos(self.planet.radius / np.linalg.norm(radius_satellite))
        angle_iteration = np.linspace(0.3, 2 * np.pi + 0.3, 360)
        latitude_footprint = np.zeros(360)
        longitude_footprint = np.zeros(360)
        for n, k in enumerate(angle_iteration):
            latitude_temp = np.rad2deg(np.arcsin(np.cos(delta_latitude) * np.sin(np.deg2rad(latitude_sat)) +
                                                 np.sin(delta_latitude) * np.cos(np.deg2rad(latitude_sat)) * np.cos(
                k)))
            sin_delta_longitude = np.sin(delta_latitude) * np.sin(k) / np.cos(np.deg2rad(latitude_temp))
            cos_delta_longitude = (np.cos(delta_latitude) - np.sin(np.deg2rad(latitude_sat))
                                   * np.sin(np.deg2rad(latitude_temp))) / (np.cos(np.deg2rad(latitude_sat))
                                                                           * np.cos(np.deg2rad(latitude_temp)))
            delta_longitude_temp = np.rad2deg(np.arctan2(sin_delta_longitude, cos_delta_longitude))

            latitude_footprint[n] = latitude_temp
            longitude_footprint[n] = delta_longitude_temp + longitude_sat

        # Longitude of footprint between -180 and 180.
        longitude_footprint = orbut.wrap_to_180(longitude_footprint)
        return latitude_footprint, longitude_footprint

    def _get_mean_anomaly(self):
        """This internal function is used to compute the mean anomaly from the true anomaly and eccentricity.

        OUTPUT:
        ------
        mean_anomaly [rad]: float
            satellite mean anomaly.
        """
        mean_and_true = orbut.MeanAnomalyTrueAnomaly(self.eccentricityNorm)
        mean_and_true.set_true_anomaly(self.trueAnomaly)
        mean_anomaly = mean_and_true.meanAnomaly
        return mean_anomaly

    def _get_period(self):
        """It computes the period of the orbit.

        OUTPUT:
        ------
        orbit_period [sec]: float
            orbit period in seconds.
        """
        orbit_period = 2 * np.pi / self.planet.gravitationalParameter ** 2 * (
                self.angularMomentumNorm / (np.sqrt(1 - self.eccentricityNorm ** 2))) ** 3
        return orbit_period

    def _get_declination_one_orbit(self):
        """This function returns the declination values during one orbit.

        OUTPUT:
        ------
        delta_vector [deg]: array_like
            vector of declination values during one orbit, in degrees.
        """
        radius = self._get_position_vector_one_orbit()
        radius_norm = np.sqrt(radius[0] ** 2 + radius[1] ** 2 + radius[2] ** 2)
        delta = np.rad2deg(np.arcsin(radius[2] / radius_norm))
        return delta

    def _get_greenwich_sideral_time(self, time):
        """This function computes the Greenwich sideral time at every time step of integration.
        """
        time_period_vector = np.linspace(0, self._get_period(), 2000)
        sideral_time_greenwich_vect = np.zeros(np.size(time_period_vector))
        for idx, k in enumerate(time_period_vector):
            ut = time.hour + time.minute / 60 + \
                 (time.second + time.microsecond * 10e-7) / 3600 + k / 3600
            stg = jd.get_greenwich_sideral_time(time.year, time.month, time.day, ut)
            sideral_time_greenwich_vect[idx] = stg
        return sideral_time_greenwich_vect

    def _get_right_ascension_one_orbit(self):
        """This function returns the right ascension values during one orbit

        OUTPUT:
        ------
        alpha_vector [deg]: array_like
            vector of right ascension values during one orbit, in degrees.
        """
        radius = self._get_position_vector_one_orbit()
        alpha_vector = np.rad2deg(np.arctan2(radius[1], radius[0]))
        return alpha_vector

    def _get_position_vector_one_orbit(self):
        """This function returns the position vector [[r_x(t)], [r_y(t)], [r_z(t)]], during one entire orbit.

        OUTPUT:
        ------
        r_vector [km]: array_like
            position vector during one entire orbit. Units are kilometers.
        """
        time_integration = np.linspace(0, self._get_period(), 2000)
        state_vector_plot = spi.odeint(orbut.state_vector_derivative,
                                       self._get_state_vector().reshape(6),
                                       time_integration, args=(self.planet.gravitationalParameter,))
        rx_vector = np.asarray([row[0] for row in state_vector_plot])
        ry_vector = np.asarray([row[1] for row in state_vector_plot])
        rz_vector = np.asarray([row[2] for row in state_vector_plot])
        r_vector = np.array([rx_vector, ry_vector, rz_vector])
        return r_vector

    @staticmethod
    def _plot_orthographic_projection(latitude_sat, longitude_sat, lat_footprint, lon_footprint, epoch):
        """This method plots the satellite track and footprint in the Orthographic projection.

        INPUT:
        ------
        latitude_sat    =   vector of latitudes of the satellite. It must be of the same size of longitude_sat. [deg]
        longitude_sat   =   vector of longitudes of the satellite. It must be of the same size of latitude_sat. [deg]
        lat_footprint   =   vector of latitudes of the boundary of the satellite footprint. [deg]
                            It must be of the same size of lon_footprint.
        lon_footprint   =   vector of longitudes of the boundary of the satellite footprint.  [deg]
                            It must be of the same size of lat_footprint.
        epoch           =   epoch to display the nightshade on the map. It must be of type datetime.
        """
        plt.figure()
        ax = plt.axes(projection=ccrs.Orthographic(longitude_sat[0], latitude_sat[0]))
        points = ccrs.Orthographic(longitude_sat[0], latitude_sat[0]).transform_points(ccrs.Geodetic(),
                                                                                       longitude_sat,
                                                                                       latitude_sat)
        footprint_points = ccrs.Orthographic(longitude_sat[0], latitude_sat[0]).transform_points(
            ccrs.Geodetic(), lon_footprint, lat_footprint)
        ax.gridlines()
        ax.coastlines()
        ax.set_global()
        ax.add_feature(Nightshade(epoch, alpha=0.4))
        ax.plot(points[:, 0], points[:, 1], linewidth=2)
        ax.plot(footprint_points[:, 0], footprint_points[:, 1], linewidth=2)
        ax.plot(points[0, 0], points[0, 1], 'h', color='#1e1596', markersize=8)

    @staticmethod
    def _plot_plate_carree_projection(latitude_sat, longitude_sat, lat_footprint, lon_footprint, epoch):
        """This method plots the satellite track and footprint in the Plate Carree projection.

        INPUT:
        ------
        latitude_sat    =   vector of latitudes of the satellite. It must be of the same size of longitude_sat.  [deg]
        longitude_sat   =   vector of longitudes of the satellite. It must be of the same size of latitude_sat. [deg]
        lat_footprint   =   vector of latitudes of the boundary of the satellite footprint. [deg]
                            It must be of the same size of lon_footprint.
        lon_footprint   =   vector of longitudes of the boundary of the satellite footprint. [deg]
                            It must be of the same size of lat_footprint.
        epoch           =   epoch to display the nightshade on the map. It must be of type datetime.
        """
        # The longitude array is cut in two piece according to positive values and negative.
        # Positive longitude values will be plotted separately from the negative one.
        longitude_positive = np.ma.masked_where(longitude_sat < 0, longitude_sat)
        longitude_negative = np.ma.masked_where(longitude_sat >= 0, longitude_sat)

        # The longitude array is cut in two piece according to positive values and negative.
        # Positive longitude values will be plotted separately from the negative one.
        longitude_footprint_positive = np.ma.masked_where(lon_footprint < 0, lon_footprint)
        longitude_footprint_negative = np.ma.masked_where(lon_footprint >= 0, lon_footprint)
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_global()
        ax.gridlines()
        ax.add_feature(Nightshade(epoch, alpha=0.4))
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(dateline_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.plot(longitude_positive, latitude_sat, color='#1f77b4', linewidth=2)
        ax.plot(longitude_negative, latitude_sat, color='#1f77b4', linewidth=2)
        ax.plot(longitude_footprint_positive, lat_footprint, color='#ff7f0e', linewidth=2)
        ax.plot(longitude_footprint_negative, lat_footprint, color='#ff7f0e', linewidth=2)
        ax.plot(longitude_sat[0], latitude_sat[0], 'h', color='#1e1596', markersize=8)
