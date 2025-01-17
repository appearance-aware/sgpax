def days2mdhms(year, days, round_to_microsecond=6):
    """Convert a float point number of days into the year into date and time.

    Given the integer year plus the "day of the year" where 1.0 means
    the beginning of January 1, 2.0 means the beginning of January 2,
    and so forth, return the Gregorian calendar month, day, hour,
    minute, and floating point seconds.

    >>> days2mdhms(2000, 1.0)   # January 1
    (1, 1, 0, 0, 0.0)
    >>> days2mdhms(2000, 32.0)  # February 1
    (2, 1, 0, 0, 0.0)
    >>> days2mdhms(2000, 366.0)  # December 31, since 2000 was a leap year
    (12, 31, 0, 0, 0.0)

    The floating point seconds are rounded to an even number of
    microseconds if ``round_to_microsecond`` is true.

    """
    second = days * 86400.0
    if round_to_microsecond:
        second = round(second, round_to_microsecond)

    minute, second = divmod(second, 60.0)
    if round_to_microsecond:
        second = round(second, round_to_microsecond)

    minute = int(minute)
    hour, minute = divmod(minute, 60)
    day_of_year, hour = divmod(hour, 24)

    is_leap = year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)
    month, day = _day_of_year_to_month_day(day_of_year, is_leap)
    if month == 13:  # behave like the original in case of overflow
        month = 12
        day += 31

    return month, day, int(hour), int(minute), second


def _day_of_year_to_month_day(day_of_year, is_leap):
    """Core logic for turning days into months, for easy testing."""
    february_bump = (2 - is_leap) * (day_of_year >= 60 + is_leap)
    august = day_of_year >= 215
    month, day = divmod(2 * (day_of_year - 1 + 30 * august + february_bump), 61)
    month += 1 - august
    day //= 2
    day += 1
    return month, day


def invjday(jd):
    #  --------------- find year and days of the year ---------------
    temp = jd - 2415019.5
    tu = temp / 365.25
    year = 1900 + int(tu // 1.0)
    leapyrs = int(((year - 1901) * 0.25) // 1.0)

    #  optional nudge by 8.64x10-7 sec to get even outputs
    days = temp - ((year - 1900) * 365.0 + leapyrs) + 0.00000000001

    #  ------------ check for case of beginning of a year -----------
    if days < 1.0:
        year = year - 1
        leapyrs = int(((year - 1901) * 0.25) // 1.0)
        days = temp - ((year - 1900) * 365.0 + leapyrs)

    #  ----------------- find remaing data  -------------------------
    mon, day, hr, minute, sec = days2mdhms(year, days, False)
    sec = sec - 0.00000086400
    return year, mon, day, hr, minute, sec


def jday(year, mon, day, hr, minute, sec):
    jd = (
        367.0 * year
        - 7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0
        + 275.0 * mon // 9.0
        + day
        + 1721013.5
    )
    fr = (sec + minute * 60.0 + hr * 3600.0) / 86400.0
    return jd, fr
