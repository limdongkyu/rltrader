import time
import datetime
import locale
import os
import platform


# Settings for Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Date Time Format
time_str = None
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"


# 로케일 설정
if 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')


# Settings on Logging
def get_today_str():
    today = datetime.datetime.combine(datetime.date.today(), datetime.datetime.min.time())
    today_str = today.strftime('%Y%m%d')
    return today_str


def get_time_str():
    global time_str
    time_str = datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME)
    return time_str
