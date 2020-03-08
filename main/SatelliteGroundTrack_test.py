import Orbit as orb
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# This file is used to verify the Orbit Classa dn its functions.

# Earth Data
radiusEarth = 6371  # [km]
muEarth = 398600  # [km^3/s^2]

# Satellite data
perigeeAltitudeSat = 1413  # [km]
apogeeAltitudeSat = 1414  # [km]
inclinationSat = np.deg2rad(51.9667)  # [deg]
eccentricitySat = 0.0000503  # [-]
rightAscensionSat = np.deg2rad(166.3503)  # [deg]
argumentPerigeeSat = np.deg2rad(97.0054)  # [deg]
trueAnomalySat = 0  # [deg]
meanAnomalySat = np.deg2rad(68.9028)  # [deg]
perigeeSat = radiusEarth + perigeeAltitudeSat
apogeeSat = radiusEarth + apogeeAltitudeSat
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
plt.show()
