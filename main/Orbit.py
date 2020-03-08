import numpy as np
import OrbitUtils
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate as spi
import datetime as dt
import julianDay as jd
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


class Orbit:
    def __init__(self, gravitational_parameter, inclination, angular_momentum_norm, right_ascension_ascending_node,
                 eccentricity_norm, argument_of_perigee, true_anomaly, epoch=dt.datetime.now()):
        """This Class allows to create an orbit around an object, specifying the six orbital elements and the epoch
        """
        # TODO: add full description of the class and its functions. epoch is datetime type.
        self.inclination = inclination
        self.angularMomentumNorm = angular_momentum_norm
        self.rightAscension = right_ascension_ascending_node
        self.eccentricityNorm = eccentricity_norm
        self.argumentOfPerigee = argument_of_perigee
        self.trueAnomaly = true_anomaly
        self.gravitationalParameter = gravitational_parameter
        self.orbitPeriod = self._get_period()
        self.meanAnomaly = self._get_mean_anomaly()
        self.epoch = epoch

    def orbital_elements2state_vector(self):
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
        radius_vector_perifocal = self.angularMomentumNorm ** 2 / (
                self.gravitationalParameter * (1 + self.eccentricityNorm * np.cos(self.trueAnomaly))) * np.array(
            [np.cos(self.trueAnomaly), np.sin(self.trueAnomaly), 0])
        velocity_vector_perifocal = self.gravitationalParameter / self.angularMomentumNorm * np.array(
            [-np.sin(self.trueAnomaly), self.eccentricityNorm + np.cos(self.trueAnomaly), 0])
        rot_matrix_perifocal2_geocentric_equat = np.array([[np.cos(self.rightAscension) * np.cos(self.argumentOfPerigee)
                                                            - np.sin(self.rightAscension) *
                                                            np.sin(self.argumentOfPerigee) * np.cos(self.inclination),
                                                            -np.cos(self.rightAscension) *
                                                            np.sin(self.argumentOfPerigee) - np.sin(self.rightAscension)
                                                            * np.cos(self.inclination) * np.cos(self.argumentOfPerigee),
                                                            np.sin(self.rightAscension) * np.sin(self.inclination)],
                                                           [np.sin(self.rightAscension) * np.cos(self.argumentOfPerigee)
                                                            + np.cos(self.rightAscension) * np.cos(self.inclination) *
                                                            np.sin(self.argumentOfPerigee), -np.sin(self.rightAscension)
                                                            * np.sin(self.argumentOfPerigee) +
                                                            np.cos(self.rightAscension) * np.cos(self.inclination) *
                                                            np.cos(self.argumentOfPerigee), -np.cos(self.rightAscension)
                                                            * np.sin(self.inclination)],
                                                           [np.sin(self.inclination) * np.sin(self.argumentOfPerigee),
                                                            np.sin(self.inclination) * np.cos(self.argumentOfPerigee),
                                                            np.cos(self.inclination)]])
        radius_vector_geocentric_equat = rot_matrix_perifocal2_geocentric_equat.dot(radius_vector_perifocal)
        velocity_vector_geocentric_equat = rot_matrix_perifocal2_geocentric_equat.dot(velocity_vector_perifocal)
        state_vector = np.array([radius_vector_geocentric_equat, velocity_vector_geocentric_equat])
        return state_vector

    def set_mean_anomaly(self, mean_anomaly):
        """This function set the mean anomaly and update the new true anomaly.

        INPUT:
        ------
        meanAnomaly = mean anomaly [rad]
        """
        self.meanAnomaly = mean_anomaly
        if mean_anomaly >= np.pi:
            eccentric_anomaly_zero = mean_anomaly - self.eccentricityNorm / 2
        else:
            eccentric_anomaly_zero = mean_anomaly + self.eccentricityNorm / 2
        eccentric_anomaly = sp.optimize.newton(self._mean_anomaly_func, eccentric_anomaly_zero,
                                               self._mean_anomaly_derivative, tol=1e-6)
        self.trueAnomaly = 2 * np.arctan(np.sqrt((1 + self.eccentricityNorm) / (1 - self.eccentricityNorm))
                                         * np.tan(eccentric_anomaly / 2))

    def plot_orbit(self):
        """This function plot the orbit in 3D.

        !!! IMPORTANT !!!:  Add at the beginning of the script:     include import matplotlib.pyplot as plt
                            Add at the end of the script:           plt.show()
        """
        # TODO: add a 3D Earth or a sphere and equal axes.
        r_plot = self._get_position_vector_one_orbit()
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(r_plot[0][:], r_plot[1][:], r_plot[2][:], 'r')
        plt.grid()

    def satellite_ground_track(self, epoch_ground_track=dt.datetime.now(), plot_map=True):
        """This function returns the latitude and longitude in one orbit of the satellite at the specified epoch.
         If the epoch is not specified, current time is consider.

         INPUT:
         ------
         epoch_ground_track =   epoch at which the ground track is plotted. Default: dt.datetime.now()
         plot_map           =   if True, two map projections are plotted with the satellite ground track and its
                                footprint. Default: True

         OUTPUT:
         -----
         latitude_satellite     = vector of latitude of the satellite during one orbit. [deg]
         longitude_satellite    = vector of longitude of the satellite during one orbit. [deg]

         !!! IMPORTANT !!!:  Add at the beginning of the script:     include import matplotlib.pyplot as plt
                             Add at the end of the script:           plt.show()
        """
        # TODO: add the minimum inclination needed to see the satellite
        # A dummy mean anomaly is computed and set in the Orbit.meanAnomaly. The real value is saved and re-set back
        # into the Orbit class.
        real_mean_anomaly = self.meanAnomaly
        dummy_mean_anomaly = self.meanAnomaly + 2 * np.pi / self.orbitPeriod * ((epoch_ground_track - self.epoch)
                                                                                .total_seconds())
        self.set_mean_anomaly(dummy_mean_anomaly)

        # Now that a mean anomaly at epoch has been set,
        # the latitude_satellite and right ascension are computed along the full orbit.
        latitude_satellite = self._get_declination_one_orbit()
        right_ascension = self._get_right_ascension_one_orbit()

        # The greenwich sideral time is computed for each time step which the orbit time has been divided in.
        greenwich_sideral_time_vector = self._get_greenwich_sideral_time(epoch_ground_track)

        # Right ascension is computed between 0 and 360 instead of -180 and 180.
        for idx, k in enumerate(right_ascension):
            if k < 0:
                right_ascension[idx] = 360 + k

        # Longitude of the Satellite.
        longitude_satellite = right_ascension - greenwich_sideral_time_vector

        # Longitude between -180 and 180.
        for idx, k in enumerate(longitude_satellite):
            if k < 0:
                longitude_satellite[idx] = 360 + k
        for idx, k in enumerate(longitude_satellite):
            if 180 < k < 360:
                longitude_satellite[idx] = -360 + k

        # Computation of the satellite footprint.
        latitude_footprint, longitude_footprint = self._get_footprint_satellite(latitude_satellite[0],
                                                                                longitude_satellite[0])
        if plot_map:
            # Plot the figure of the track with the map on the Orthographic and PlateCarree.
            self._plot_orthographic_projection(latitude_satellite, longitude_satellite,
                                               latitude_footprint, longitude_footprint, epoch_ground_track)
            self._plot_plate_carree_projection(latitude_satellite, longitude_satellite,
                                               latitude_footprint, longitude_footprint, epoch_ground_track)

        # The mean anomaly is set back to its actual value.
        self.set_mean_anomaly(real_mean_anomaly)

        return latitude_satellite, longitude_satellite

    def get_right_ascension(self):
        """This function returns the right ascension of the satellite.

        OUTPUT:
        ------
        alpha = right ascension of the satellite [deg]
        """
        r = self.orbital_elements2state_vector()[0][:]
        alpha = np.rad2deg(np.arctan2(r[1], r[0]))
        return alpha

    def get_declination(self):
        """This function returns the declination of the satellite.

        OUTPUT:
        ------
        delta = declination of the satellite [deg]
        """
        r = self.orbital_elements2state_vector()[0][:]
        radius_norm = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        delta = np.rad2deg(np.arcsin(r[2] / radius_norm))
        return delta

    def is_satellite_visible(self, station_latitude, station_longitude, epoch_ground_station=dt.datetime.now(),
                             plot_map=True):
        """This method verifies if a satellite is visible from a ground station at epoch.

        INPUT:
        ------
        observer_latitude        =   latitude of the ground station.   [deg]
        observer_longitude       =   longitude of the ground station.  [deg]
        epoch_ground_station    =   epoch of interest.
        plot_map                =   if True, two map projections are plotted showing the satellite ground track,
                                    its footprint and the ground station position. Default: True

        OUTPUT:
        ------
        station_visible         =   True if  the satellite is visible from the ground station at epoch.
                                    False if the satellite is NOT visible from the ground station at epoch.

        !!! IMPORTANT !!!:  Add at the beginning of the script:     include import matplotlib.pyplot as plt
                             Add at the end of the script:           plt.show()
        """
        # The ground track is first computed and plotted.
        latitude_sat, longitude_sat = self.satellite_ground_track(epoch_ground_track=epoch_ground_station, plot_map=plot_map)
        if plot_map:
            # Number Id of the figures opened by the method self.plot_ground_track is retrieved.
            plate_carree_figure_number = plt.gcf().number
            orthographic_figure_number = plate_carree_figure_number - 1
        # A dummy mean anomaly is computed and set in the Orbit.meanAnomaly. The real value is saved and re-set back
        # into the Orbit class at the end of the method.
        real_mean_anomaly = self.meanAnomaly
        dummy_mean_anomaly = self.meanAnomaly + 2 * np.pi / self.orbitPeriod * ((epoch_ground_station - self.epoch)
                                                                                .total_seconds())
        self.set_mean_anomaly(dummy_mean_anomaly)

        satellite_current_declination = self.get_declination()
        satellite_current_right_ascension = self.get_right_ascension()

        if satellite_current_right_ascension < 0:
            satellite_current_right_ascension += 360

        # The greenwich sideral time at epoch is computed.
        greenwich_sideral_time = self._get_greenwich_sideral_time(epoch_ground_station)[0]

        # Longitude of the Satellite.
        satellite_current_longitude = satellite_current_right_ascension - greenwich_sideral_time

        # Longitude between -180 and 180.
        if satellite_current_longitude < 0:
            satellite_current_longitude += 360
        elif 180 < satellite_current_longitude < 360:
            satellite_current_longitude -= 360

        delta_longitude_satellite_station = np.abs(satellite_current_longitude - station_longitude)

        # Conversion of some variables from deg to rad.
        satellite_current_declination_rad = np.deg2rad(satellite_current_declination)
        station_latitude_rad = np.deg2rad(station_latitude)
        delta_longitude_sat_station_rad = np.deg2rad(delta_longitude_satellite_station)

        radius_satellite = self.orbital_elements2state_vector()[0:3]
        # TODO: the radius of the Earth should be an input of the Orbit class
        max_central_angle = np.arccos(6371 / np.linalg.norm(radius_satellite))
        station_satellite_angle_on_geodetic = np.arccos(
            np.sin(satellite_current_declination_rad) * np.sin(station_latitude_rad) +
            np.cos(satellite_current_declination_rad) * np.cos(station_latitude_rad) *
            np.cos(delta_longitude_sat_station_rad))

        azimuth, elevation = OrbitUtils.object_sky_locator(latitude_sat, longitude_sat, station_latitude,
                                                           station_latitude)
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
        self.set_mean_anomaly(real_mean_anomaly)
        return station_visible

    def _get_footprint_satellite(self, latitude_sat, longitude_sat):
        """This function returns the two vectors of latitude and associated longitude for the boundary of the satellite
        footprint, given the current latitude and longitude of the satellite.

        INPUT:
            ------
            latitude_sat    = Current latitude of the satellite [deg]
            longitude_sat   = Current longitude of the satellite [deg]

            OUTPUT:
            -------
            latitude_footprint		= vector of latitudes of the boundary of the satellite footprint [deg]
            longitude_footprint		= vector of longitudes of the boundary of the satellite footprint [deg]"""
        radius_satellite = self.orbital_elements2state_vector()[0:3]
        # delta_latitude is the latitude, which added to the current satellite latitude,
        # gives the max latitude of visibility.
        # TODO: the radius of the Earth should be an input of the Orbit class
        delta_latitude = np.arccos(6371 / np.linalg.norm(radius_satellite))
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
        for idx, k in enumerate(longitude_footprint):
            if k < -180:
                longitude_footprint[idx] = 360 + k
        for idx, k in enumerate(longitude_footprint):
            if 180 < k < 360:
                longitude_footprint[idx] = -360 + k

        return latitude_footprint, longitude_footprint

    def _get_eccentric_anomaly(self):
        eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - self.eccentricityNorm) / (1 + self.eccentricityNorm))
                                          * np.tan(self.trueAnomaly / 2))
        return eccentric_anomaly

    def _get_mean_anomaly(self):
        ecc_anomaly = self._get_eccentric_anomaly()
        mean_anomaly = ecc_anomaly - self.eccentricityNorm * np.sin(ecc_anomaly)
        return mean_anomaly

    def _get_period(self):
        """It computes the period of the orbit.

        OUTPUT:
        ------
        orbit_period = orbit period in seconds. [sec]
        """
        orbit_period = 2 * np.pi / self.gravitationalParameter ** 2 * (
                self.angularMomentumNorm / (np.sqrt(1 - self.eccentricityNorm ** 2))) ** 3
        return orbit_period

    def _get_declination_one_orbit(self):
        """This function returns the declination values during one orbit.

        OUTPUT:
        ------
        delta_vector = vector of declination values during one orbit. [deg]
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
        alpha_vector = vector of right ascension values during one orbit. [deg]
        """
        radius = self._get_position_vector_one_orbit()
        alpha_vector = np.rad2deg(np.arctan2(radius[1], radius[0]))
        return alpha_vector

    def _mean_anomaly_func(self, eccentric_anomaly):
        mean_anomaly = eccentric_anomaly - self.eccentricityNorm * np.sin(eccentric_anomaly) - self.meanAnomaly
        return mean_anomaly

    def _mean_anomaly_derivative(self, eccentric_anomaly):
        mean_anomaly_derivative = 1 - self.eccentricityNorm * np.cos(eccentric_anomaly)
        return mean_anomaly_derivative

    def _get_position_vector_one_orbit(self):
        """This function returns the position vector [[r_x(t)], [r_y(t)], [r_z(t)]], during one entire orbit.

        OUTPUT:
        ------
        r_vector = position vector during one entire orbit. [km]
        """
        time_integration = np.linspace(0, self._get_period(), 2000)
        state_vector_plot = spi.odeint(OrbitUtils.state_vector_derivative,
                                       self.orbital_elements2state_vector().reshape(6),
                                       time_integration, args=(self.gravitationalParameter,))
        rx_vector = np.asarray([row[0] for row in state_vector_plot])
        ry_vector = np.asarray([row[1] for row in state_vector_plot])
        rz_vector = np.asarray([row[2] for row in state_vector_plot])
        r_vector = np.array([rx_vector, ry_vector, rz_vector])
        return r_vector

    # TODO: should this and the following method be a function of OrbitUtils?
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
