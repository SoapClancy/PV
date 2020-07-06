import os
import pandas as pd
import re
from PVPanel_Class import PVPanel
from project_path_Var import project_path_
from Ploting.fast_plot_Func import series, hist
from Time_Processing.format_convert_Func import datetime_to_mktime
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates

MANUFACTURER = ('Sanyo', 'STP', 'DF', 'Yingli')
CONFIGURATION = ('open', 'closed', 'tracker')


def load_raw_data(raw_data_excel_path=project_path_ + 'Data/Raw_measurements/',
                  file_name="pvlog.2018-03-29") -> pd.DataFrame:
    if not os.path.exists(raw_data_excel_path + file_name + ".pkl"):
        read_results = pd.read_csv(raw_data_excel_path + file_name + ".csv", sep=';')  # type: pd.DataFrame
        read_results['ID'].astype('int')
        read_results['Time'] = pd.to_datetime(read_results['Time'])
        read_results.to_pickle(raw_data_excel_path + file_name + ".pkl")  # type: pd.DataFrame
    else:
        read_results = pd.read_pickle(raw_data_excel_path + file_name + ".pkl")  # type: pd.DataFrame

    return read_results


def initialise_pv_using_raw_data_and_then_filter():
    # %% 载入原始数据
    raw_data = load_raw_data()

    # %% 初始化所有的PVPanel实例
    # 设置所有的PVPanel实例的初值(空字典{})
    all_pv = {}.fromkeys(MANUFACTURER)

    # 把共同都需要的记录提取出来(ID, time, environmental temperature, fixed irradiation, tracker irradiation, wind speed)
    common_measurements = raw_data[['ID', 'Time', 'T_Omgeving', 'Solar_Meter fixed',
                                    'Solar_Meter Dubbel Tracker', 'Windspeed']]  # type: pd.DataFrame
    common_measurements = pd.concat([common_measurements,
                                     PVPanel.cal_recording_rank_in_a_day(common_measurements['Time'])],
                                    axis=1)
    common_measurements = common_measurements.rename(columns={'Time': 'time',
                                                              'T_Omgeving': 'environmental temperature',
                                                              'Solar_Meter fixed': 'fixed irradiation',
                                                              'Solar_Meter Dubbel Tracker': 'tracker irradiation',
                                                              'Windspeed': 'wind speed'})  # type: pd.DataFrame

    # 为所有的PVPanel实例赋值
    for this_manufacturer in MANUFACTURER:
        all_pv[this_manufacturer] = {}
        for this_configuration in CONFIGURATION:
            col_idx = [x for x in raw_data.columns if re.search(this_manufacturer, x, re.I)]
            col_idx = [x for x in col_idx if re.search(this_configuration, x, re.I)]
            if not col_idx:
                all_pv[this_manufacturer][this_configuration] = None
                continue
            else:
                power_idx = [x for x in col_idx if re.search('P_', x, re.I)][0]
                temperature_idx = [x for x in col_idx if re.search('T_', x, re.I)][0]
                measurements = pd.DataFrame({'power output': raw_data[power_idx],
                                             'panel temperature': raw_data[temperature_idx]})
                measurements = pd.concat([common_measurements, measurements], axis=1)
                all_pv[this_manufacturer][this_configuration] = PVPanel(manufacturer=this_manufacturer,
                                                                        configuration=this_configuration,
                                                                        measurements=measurements)
    return all_pv


def check_data_availability_from_raw_data():
    # %% 载入原始数据
    raw_data = load_raw_data()
    time_data = raw_data['Time'].values
    hist(time_data, bins=9*12, x_label='Year', y_label='Frequency')


if __name__ == '__main__':
    check_data_availability_from_raw_data()
