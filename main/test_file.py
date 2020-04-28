import Orbit as orb
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from SolarSystem import Earth, Mars

# **** This file is used to verify the new methods of the Class Orbit and the SolarSystem module ****

# TEST OF THE SOLAR SYSTEM MODULE
# Using EXAMPLE 8.7 of H. Curtis - Orbital Mechanics for Engineering Students, Third Edition.
day = dt.datetime(2003, 8, 27, 12, 0, 0)
Earth.set_new_epoch(day)
Mars.set_new_epoch(day)

print("Earth position: ", Earth.position)   # Earth z-position differs from Curtis by its sign. It's a Curtis mistake!
print("Earth velocity: ", Earth.velocity)
print("Mars position: ", Mars.position)
print("Mars velocity: ", Mars.velocity)

distance_Earth_Mars = np.linalg.norm(Earth.position - Mars.position)

print("\nEarth-Mars distance on", Earth.epoch, ": ", distance_Earth_Mars, "km")


# TEST OF THE ORBIT MODULE
# Earth Data
muEarth = Earth.gravitationalParameter  # [km^3/s^2]

# Satellite data
perigeeAltitudeSat = 1413  # [km]
apogeeAltitudeSat = 1414  # [km]
inclinationSat = np.deg2rad(51.9667)  # [deg]
eccentricitySat = 0.0000503  # [-]
rightAscensionSat = np.deg2rad(166.3503)  # [deg]
argumentPerigeeSat = np.deg2rad(97.0054)  # [deg]
trueAnomalySat = 0  # [deg]
meanAnomalySat = np.deg2rad(68.9028)  # [deg]
perigeeSat = Earth.radius + perigeeAltitudeSat
apogeeSat = Earth.radius + apogeeAltitudeSat
angularMomentumSat = np.sqrt(apogeeSat * muEarth * (1 - eccentricitySat))

# Satellite Epoch
yearSat = 2020
monthSat = 2
daySat = 15
hourSat = 19
minuteSat = 25
secondSat = 4
timeNow = dt.datetime.utcnow()
timeEpoch = dt.datetime(yearSat, monthSat, daySat, hourSat, minuteSat, secondSat)

# Satellite orbit definition and parameters
Satellite = orb.Orbit(muEarth, inclinationSat, angularMomentumSat, rightAscensionSat, eccentricitySat,
                      argumentPerigeeSat, trueAnomalySat, epoch=timeEpoch)
Satellite.set_mean_anomaly(meanAnomalySat)
lat, lon = Satellite.satellite_ground_track()
isvisible = Satellite.is_satellite_visible(60, 120)
# plt.show()
