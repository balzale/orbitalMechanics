import Orbit as orb
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from SolarSystem import Earth, Mars
import OrbitUtils as orbut

# **** This file is used to verify the new methods of the Class Orbit and the SolarSystem module ****

# TEST OF THE SOLAR SYSTEM MODULE
# Using EXAMPLE 8.7 of H. Curtis - Orbital Mechanics for Engineering Students, Third Edition.

# day = dt.datetime(2003, 8, 27, 12, 0, 0)
# Earth.set_new_epoch(day)
# Mars.set_new_epoch(day)
#
# print("Earth position: ", Earth.position)   # Earth z-position differs from Curtis by its sign. It's a Curtis mistake!
# print("Earth velocity: ", Earth.velocity)
# print("Mars position: ", Mars.position)
# print("Mars velocity: ", Mars.velocity)
#
# distance_Earth_Mars = np.linalg.norm(Earth.position - Mars.position)
#
# print("\nEarth-Mars distance on", Earth.epoch, ": ", distance_Earth_Mars, "km")


# TEST OF THE ORBIT MODULE
# Earth Data
muEarth = Earth.gravitationalParameter  # [km^3/s^2]

# Satellite data
perigeeAltitudeSat = 23215  # [km]
apogeeAltitudeSat = 23228  # [km]
inclinationSat = np.deg2rad(56.3287)  # [deg]
eccentricitySat = 0.0002168  # [-]
rightAscensionSat = np.deg2rad(283.0922)  # [deg]
argumentPerigeeSat = np.deg2rad(340.0164)  # [deg]
meanAnomalySat = np.deg2rad(19.9239)  # [deg]
perigeeSat = Earth.radius + perigeeAltitudeSat
apogeeSat = Earth.radius + apogeeAltitudeSat
angularMomentumSat = np.sqrt(apogeeSat * muEarth * (1 - eccentricitySat))
mean_to_true = orbut.MeanAnomalyTrueAnomaly(eccentricitySat)
mean_to_true.set_mean_anomaly(meanAnomalySat)
trueAnomalySat = mean_to_true.trueAnomaly

# Satellite Epoch
yearSat = 2020
monthSat = 5
daySat = 6
hourSat = 20
minuteSat = 14
secondSat = 35
timeNow = dt.datetime.now()
timeEpoch = dt.datetime(yearSat, monthSat, daySat, hourSat, minuteSat, secondSat)

# Satellite orbit definition and parameters
Satellite = orb.Orbit(Earth, inclinationSat, angularMomentumSat, rightAscensionSat, eccentricitySat,
                      argumentPerigeeSat, trueAnomalySat, timeEpoch)
Satellite.set_new_epoch(dt.datetime.now())
lat, lon = Satellite.satellite_ground_track()
# isvisible = Satellite.is_satellite_visible(60, 120)
# Satellite.plot_orbit_3d()
Satellite.plot_orbit_perifocal()
print("Satellite in light: ", Satellite.is_satellite_in_light())
plt.show()
