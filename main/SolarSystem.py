import datetime as dt
import julianDay as jd
import numpy as np
import OrbitUtils as orbut


class Sun:
    """
    Sun class.

    Attributes
    ----------
    gravitationalParameter : float
        gravitational parameter of the Sun [km^3/s^2]
    radius : float
        radius of the Sun [km]
    mass : float
        mass of the Sun [km]
    """
    name = "Sun"
    gravitationalParameter = 132712000000.0
    radius = 696000.0
    mass = 1.989*1e30


class _Planet:
    """
    This Class allows the generation of the nine planets of the Solar System, ad allows to get the planet parameters
    according to a desired epoch.

    Attributes
    ----------
    name : str
        name of the planet
    gravitationalParameter : float
        gravitational parameter [km^3/s^2]
    mass : float
        mass of the planet [kg]
    radius : float
        radius of the planet [km]
    epoch : datetime
        epoch at which data is computed
    semimajorAxis: float
        semimajor axis of the planet orbit [km]
    eccentricity: float
        eccentricity of the planet orbit [-]
    angularMomentumNorm: float
        norm of the angular momentum of the planet orbit [km^2/s]
    inclination: float
        inclination of the planet orbit with respect to the ecliptic [deg]
    rightAscension: float
        right ascension of the ascending node of the planet orbit [deg]
    argumentOfPerihelion: float
        argument of the perihelion of the planet orbit [deg]
    meanAnomaly: float
        mean anomaly at epoch of the planet [deg]
    trueAnomaly: float
        true anomaly at epoch of the planet [deg]
    position: array_like
        array 3x1 [r_x, r_y, r_z] with the planet positions at epoch in the Celestial inertial frame. [km]
    velocity: array_like
        array 3x1 [v_x, v_y, v_z] with the planet velocities at epoch in the Celestial inertial frame [km]

    Methods
    -------
    set_new_epoch(new_epoch=dt.datetime.now())
        Sets the desired epoch for the planet and updates all the attributes.
    """
    def __init__(self, planet_id):
        """
        Parameters:
        ----------
        planet_id: int
            Id of the planet, as follow:

            1: Mercury
            2: Venus
            3: Earth
            4: Mars
            5: Jupiter
            6: Saturn
            7: Uranus
            8: Neptune
            9: Pluto
        """
        self._id = planet_id
        self.name = self._get_planet_name()
        self.gravitationalParameter = self._get_gravitational_parameter()
        self.mass = self._get_mass()
        self.radius = self._get_radius()
        self.epoch = dt.datetime.now()
        self._julianDay = jd.julian_day(self.epoch.year, self.epoch.month, self.epoch.day, self.epoch.hour,
                                        self.epoch.minute, self.epoch.second)
        self.semimajorAxis = self._get_semimajor_axis()
        self.eccentricity = self._get_eccentricity()
        self.angularMomentumNorm = self._get_angular_momentum_norm()
        self.inclination = self._get_inclination()
        self.rightAscension = self._get_right_ascension_ascending_node()
        self._longitudeOfPerihelion = self._get_longitude_of_perihelion()
        self.argumentOfPerihelion = self._get_argument_of_perihelion()
        self._meanLongitude = self._get_mean_longitude()
        self.meanAnomaly = self._get_mean_anomaly()
        self.trueAnomaly = self._get_true_anomaly()
        self.position = self._get_state_vector()[0]
        self.velocity = self._get_state_vector()[1]

    def set_new_epoch(self, new_epoch=dt.datetime.now()):
        """This function sets the desired epoch for the planet and updates all the attributes.

        INPUT:
        -------
        new_epoch: datetime
            the new epoch desired of the planet (Default: current date and time)
        """
        self.epoch = new_epoch
        self._julianDay = jd.julian_day(self.epoch.year, self.epoch.month, self.epoch.day, self.epoch.hour,
                                        self.epoch.minute, self.epoch.second)
        self.semimajorAxis = self._get_semimajor_axis()
        self.eccentricity = self._get_eccentricity()
        self.angularMomentumNorm = self._get_angular_momentum_norm()
        self.inclination = self._get_inclination()
        self.rightAscension = self._get_right_ascension_ascending_node()
        self._longitudeOfPerihelion = self._get_longitude_of_perihelion()
        self.argumentOfPerihelion = self._get_argument_of_perihelion()
        self._meanLongitude = self._get_mean_longitude()
        self.meanAnomaly = self._get_mean_anomaly()
        self.trueAnomaly = self._get_true_anomaly()
        self.position = self._get_state_vector()[0]
        self.velocity = self._get_state_vector()[1]

    def _get_gravitational_parameter(self):
        """This function returns the Gravitational Parameter (mu) of the planet chosen and the epoch.

        Data from from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition, Table A.2

        OUTPUT:
        -------
        planet Gravitational Parameter [km^3/s^2]: float
        """
        switcher = {
            1: 22030.0,       # Mercury
            2: 324900.0,      # Venus
            3: 398600.0,      # Earth
            4: 42828.0,       # Mars
            5: 126686000.0,   # Jupiter
            6: 37931000.0,    # Saturn
            7: 5794000.0,     # Uranus
            8: 6835100.0,     # Neptune
            9: 830.0          # Pluto
        }
        return switcher.get(self._id, float('nan'))

    def _get_planet_name(self):
        """This function returns the planet name, according to the Id selected during Planet class creation.

        OUTPUT:
        -------
        planet name: char
        """
        switcher = {
            1: "Mercury",
            2: "Venus",
            3: "Earth",
            4: "Mars",
            5: "Jupiter",
            6: "Saturn",
            7: "Uranus",
            8: "Neptune",
            9: "Pluto"
        }
        return switcher.get(self._id, "Unknown")

    def _get_mass(self):
        """This function returns the mass of the planet chosen and the epoch.

        Data from from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition, Table A.1

        OUTPUT:
        -------
        planet mass [kg]: float
        """
        switcher = {
            1: 330.2 * 1e21,    # Mercury
            2: 4.869 * 1e24,    # Venus
            3: 5.974 * 1e24,    # Earth
            4: 641.9 * 1e21,    # Mars
            5: 1.899 * 1e27,    # Jupiter
            6: 568.5 * 1e24,    # Saturn
            7: 86.83 * 1e24,    # Uranus
            8: 102.4 * 1e24,    # Neptune
            9: 12.50 * 1e21     # Pluto
        }
        return switcher.get(self._id, float('nan'))

    def _get_radius(self):
        """This function returns the value of the radius according to the planet chosen and the epoch.

        Data from from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition, Table A.1

        OUTPUT:
        -------
        planet radius [km]: float
        """
        switcher = {
            1: 2440.0,      # Mercury
            2: 6052.0,      # Venus
            3: 6378.0,      # Earth
            4: 3396.0,      # Mars
            5: 71490.0,     # Jupiter
            6: 60270.0,     # Saturn
            7: 25560.0,     # Uranus
            8: 24760.0,     # Neptune
            9: 1195.0       # Pluto
        }
        return switcher.get(self._id, float('nan'))

    def _get_semimajor_axis(self):
        """This function returns the value of the semimajor axis according to the planet chosen and the epoch.

        The logic and data are taken from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition,
        Eq. 8.93b & Table 8.1.

        OUTPUT:
        -------
        semimajor axis [km]: float
        """
        t0 = jd.t_zero(self._julianDay)
        switcher = {
            1: 0.387099270 + t0 * 0.00000037,   # Mercury
            2: 0.723335660 + t0 * 0.00000390,   # Venus
            3: 1.000002610 + t0 * 0.00000562,   # Earth
            4: 1.523710340 + t0 * 0.0001847,    # Mars
            5: 5.202887000 + t0 * -0.00011607,  # Jupiter
            6: 9.536675940 + t0 * -0.00125060,  # Saturn
            7: 19.18916464 + t0 * -0.00196176,  # Uranus
            8: 30.06992276 + t0 * 0.00026291,   # Neptune
            9: 39.48211675 + t0 * -0.00031596   # Pluto
        }
        return switcher.get(self._id, float('nan')) * 1.49597871*1e8

    def _get_eccentricity(self):
        """This function returns the value of the eccentricity according to the planet chosen and the epoch.

        The logic and data are taken from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition,
        Eq. 8.93b & Table 8.1.

        OUTPUT:
        -------
        eccentricity [-]: float
        """
        t0 = jd.t_zero(self._julianDay)
        switcher = {
            1: 0.20563593 + t0 * 0.00001906,    # Mercury
            2: 0.00677672 + t0 * -0.00004107,   # Venus
            3: 0.01671123 + t0 * -0.00004392,   # Earth
            4: 0.09339410 + t0 * 0.00007882,    # Mars
            5: 0.04838624 + t0 * 0.00013253,    # Jupiter
            6: 0.05386179 + t0 * -0.00050991,   # Saturn
            7: 0.04725744 + t0 * -0.00004397,   # Uranus
            8: 0.00859048 + t0 * 0.00005105,    # Neptune
            9: 0.24882730 + t0 * 0.00005170     # Pluto
        }
        return switcher.get(self._id, float('nan'))

    def _get_angular_momentum_norm(self):
        """This function returns the value of the angular momentum norm.

        The angular momentum norm (h) is computed as Eq. 2.71, H. Curtis - Orbital Mechanics for Engineering Students,
        Third Edition.

        OUTPUT:
        -------
        angular momentum norm [km^2/s]: float
        """
        angular_momentum_norm = np.sqrt(Sun.gravitationalParameter * self.semimajorAxis * (1 - self.eccentricity**2))
        return angular_momentum_norm

    def _get_inclination(self):
        """This function returns the value of the inclination according to the planet chosen and the epoch.
        The value of inclination returned is in degrees and between 0-360°.

        The logic and data are taken from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition,
        Eq. 8.93b & Table 8.1

        OUTPUT:
        -------
        inclination [deg]: float
        """
        t0 = jd.t_zero(self._julianDay)
        switcher = {
            1:  7.00497902 + t0 * -0.00594749,  # Mercury
            2:  3.39467605 + t0 * -0.00078890,  # Venus
            3: -0.00001531 + t0 * -0.01294668,  # Earth
            4:  1.84969142 + t0 * -0.00813131,  # Mars
            5:  1.30439695 + t0 * -0.00183714,  # Jupiter
            6:  2.48599187 + t0 * 0.00193609,   # Saturn
            7:  0.77263783 + t0 * -0.00242939,  # Uranus
            8:  1.77004347 + t0 * 0.00035372,   # Neptune
            9: 17.14001206 + t0 * 0.00004818    # Pluto
        }
        return switcher.get(self._id, float('nan')) % 360

    def _get_right_ascension_ascending_node(self):
        """This function returns the value of the right ascension according to the planet chosen and the epoch.
        The value returned is in degrees and between 0-360°.

        The logic and data are taken from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition,
        Eq. 8.93b & Table 8.1

        OUTPUT:
        -------
        right ascension of the ascending node [deg]: float
        """
        t0 = jd.t_zero(self._julianDay)
        switcher = {
            1:  48.33076593 + t0 * -0.12534081,     # Mercury
            2:  76.67984255 + t0 * -0.27769418,     # Venus
            3:          0.0 + t0 * 0.0,             # Earth
            4:  49.55953891 + t0 * -0.29257343,     # Mars
            5: 100.47390909 + t0 * 0.20469106,      # Jupiter
            6: 113.66242448 + t0 * 0.28867794,      # Saturn
            7:  74.01692503 + t0 * 0.04240589,      # Uranus
            8: 131.78422574 + t0 * -0.00508664,     # Neptune
            9: 110.30393684 + t0 * -0.01183482      # Pluto
        }
        return switcher.get(self._id, float('nan')) % 360

    def _get_longitude_of_perihelion(self):
        """This function returns the longitude of the perihelion according to the planet chosen and the epoch.
        The value returned is in degrees and between 0-360°.

        The longitude of the perihelion is defined as the sum of the right ascension of the ascending node and
        the argument of the perihelion.

        The logic and data are taken from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition,
        Eq. 8.93b & Table 8.1

        OUTPUT:
        -------
        longitude of the perihelion [deg]: float
        """
        t0 = jd.t_zero(self._julianDay)
        switcher = {
            1:  77.45779628 + t0 * 0.16047689,      # Mercury
            2: 131.60246718 + t0 * 0.00268329,      # Venus
            3: 102.93768193 + t0 * 0.32327364,      # Earth
            4: -23.94362959 + t0 * 0.44441088,      # Mars
            5:  14.72847983 + t0 * 0.21252668,      # Jupiter
            6:  92.59887831 + t0 * -0.41897216,     # Saturn
            7: 170.95427630 + t0 * 0.40805281,      # Uranus
            8:  44.96476227 + t0 * -0.32241464,     # Neptune
            9: 224.06891629 + t0 * -0.04062942      # Pluto
        }
        return switcher.get(self._id, float('nan')) % 360

    def _get_argument_of_perihelion(self):
        """This function returns the argument of the perihelion according to the planet chosen and the epoch.
        The value returned is in degrees and between 0-360°.

        The argument of the perihelion is computed as the difference between the longitude of the perihelion and
        the right ascension of the ascending node.

        The logic and data are taken from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition,
        Eq. 8.93b & Table 8.1

        OUTPUT:
        -------
        argument of the perihelion [deg]: float
        """
        argument_of_perihelion = (self._longitudeOfPerihelion - self.rightAscension) % 360
        return argument_of_perihelion

    def _get_mean_longitude(self):
        """This function returns the mean longitude according to the planet chosen and the epoch.
        The value returned is in degrees and between 0-360°.

        The mean longitude is defined as the sum of the longitude of the perihelion and the mean anomaly.

        The logic and data are taken from H. Curtis - Orbital Mechanics for Engineering Students, Third Edition,
        Eq. 8.93b & Table 8.1

        OUTPUT:
        -------
        mean longitude [deg]: float
        """
        t0 = jd.t_zero(self._julianDay)
        switcher = {
            1: 252.25032350 + t0 * 149472.67411175,     # Mercury
            2: 181.97909950 + t0 * 58517.81538729,      # Venus
            3: 100.46457166 + t0 * 35999.37244981,      # Earth
            4: -4.553432050 + t0 * 19140.30268499,      # Mars
            5:  34.39644501 + t0 * 3034.74612775,       # Jupiter
            6:  49.95424423 + t0 * 1222.49362201,       # Saturn
            7: 313.23810451 + t0 * 428.48202785,        # Uranus
            8: -55.12002969 + t0 * 218.45945325,        # Neptune
            9: 238.92903833 + t0 * 145.20780515         # Pluto
        }
        return switcher.get(self._id, float('nan')) % 360

    def _get_mean_anomaly(self):
        """This function returns the mean anomaly at epoch.
        The value returned is in degrees and between 0-360°.

        The mean anomaly is computed as the difference between of the mean longitude and the longitude of the
        perihelion.

        OUTPUT:
        -------
        mean anomaly [deg]: float
        """
        mean_anomaly = (self._meanLongitude - self._longitudeOfPerihelion) % 360
        return mean_anomaly

    def _get_true_anomaly(self):
        """This function returns the true anomaly at epoch.
        The value returned is in degrees and between 0-360°

        OUTPUT:
        -------
        true anomaly [deg]: float
        """
        true_and_mean = orbut.MeanAnomalyTrueAnomaly(self.eccentricity)
        true_and_mean.set_mean_anomaly(np.deg2rad(self.meanAnomaly))
        return np.rad2deg(true_and_mean.trueAnomaly) % 360

    def _get_state_vector(self):
        inclination_rad = np.deg2rad(self.inclination)
        argument_perihelion_rad = np.deg2rad(self.argumentOfPerihelion)
        right_ascension_node_rad = np.deg2rad(self.rightAscension)
        true_anomaly_rad = np.deg2rad(self.trueAnomaly)
        state_vector = orbut.orbital_elements2state_vector(inclination_rad, self.angularMomentumNorm,
                                                           right_ascension_node_rad, self.eccentricity,
                                                           argument_perihelion_rad, true_anomaly_rad,
                                                           Sun.gravitationalParameter)
        return state_vector


Mercury = _Planet(1)
Venus = _Planet(2)
Earth = _Planet(3)
Mars = _Planet(4)
Jupiter = _Planet(5)
Saturn = _Planet(6)
Uranus = _Planet(7)
Neptune = _Planet(8)
Pluto = _Planet(9)
