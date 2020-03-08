import numpy as np


def j0(year, month, day):
    """Julian day number at 0 h universal_time as defined in H. Curtis - Orbital Mechanics for Engineering Students
    by equation (5.48).

    INPUT:
    ------
    year    = year of interest
    month   = month of interest
    day     = day of interest

    OUTPUT:
    ------
    j_zero   = Julian day number at 0 h universal_time
    """
    j_zero = 367*year - np.trunc((7*(year+np.trunc((month+9)/12)))/4) + np.trunc(275*month/9) + day + 1721013.5
    return j_zero


def julian_day(year, month, day, hours, minutes, seconds):
    """This function returns the Julian day number as defined in H. Curtis - Orbital Mechanics for Engineering Students
    by equation (5.47).

    INPUT:
    ------
    year    = year of the date of interest
    month   = month of the date interest
    day     = day of the date interest
    hours   = hours of the date interest
    minutes = minutes of the date interest
    seconds = seconds of the date interest

    OUTPUT:
    ------
    jd = julian day
    """
    ut = hours + minutes/60 + seconds/3600
    jd = j0(year, month, day) + ut/24
    return jd


def get_greenwich_sideral_time(year, month, day, universal_time):
    """This function returns the Greenwich sideral time (GST) at the date of interest.
    The algorithm to compute the GST is taken from H. Curtis - Orbital Mechanics for Engineering Students,
    Chapter - Preliminary Orbit determination, in section Sideral Time.

    INPUT:
    ------
    year        = year of the date of interest
    month       = month of the date of interest
    day         = day of the date of interest
    universal_time          = universal time of interest - [hours]

    OUTPUT:
    ------
    greenwich_sideral_time = greenwich sideral time at the date of interest"""
    tzero = (j0(year, month, day) - 2451545)/36525
    theta_gzero = 100.4606184 + 36000.77004 * tzero + 0.000387933 * tzero ** 2 - 2.583 * 1e-8 * tzero ** 3
    # universal_time = hours + minutes / 60 + seconds / 3600
    greenwich_sideral_time_first = theta_gzero + 360.98564724*(universal_time / 24)
    greenwich_sideral_time = greenwich_sideral_time_first - np.trunc(greenwich_sideral_time_first/360)*360
    return greenwich_sideral_time
