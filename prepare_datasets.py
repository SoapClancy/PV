import os
import pandas as pd
import re
from PVPanel_Class import PVPanel
from project_utils import project_path_
from Ploting.fast_plot_Func import series, hist
from Time_Processing.format_convert_Func import datetime_to_mktime
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
from collections import ChainMap
import copy
from File_Management.load_save_Func import *

MANUFACTURER = ('Sanyo', 'STP', 'DF', 'Yingli')
CONFIGURATION = ('open', 'closed', 'tracker')


def load_raw_data(raw_data_excel_path=project_path_ / 'Data/Raw_measurements/',
                  file_name="pvlog.2014-06-23") -> pd.DataFrame:
    raw_data_excel_path = raw_data_excel_path.__str__() + '/'
    if not os.path.exists(raw_data_excel_path + file_name + ".pkl"):
        read_results = pd.read_csv(raw_data_excel_path + file_name + ".csv", sep=';')  # type: pd.DataFrame
        read_results['ID'].astype('int')
        read_results['Time'] = pd.to_datetime(read_results['Time'])
        read_results.to_pickle(raw_data_excel_path + file_name + ".pkl")  # type: pd.DataFrame
    else:
        read_results = pd.read_pickle(raw_data_excel_path + file_name + ".pkl")  # type: pd.DataFrame

    return read_results


def initialise_pv_using_raw_data(try_to_filer_using_2019_results: bool = False):
    # %% 载入原始数据
    raw_data = load_raw_data()
    raw_data.set_index('Time', inplace=True)

    common_columns = ['ID', 'T_Omgeving', 'Solar_Meter fixed', 'Solar_Meter Dubbel Tracker', 'Windspeed']
    rename_mapper = {
        'Solar_Meter fixed': 'fixed irradiation',
        'Solar_Meter Dubbel Tracker': 'tracker irradiation',
        'Windspeed': 'wind speed',
        'T_Omgeving': 'environmental temperature'
    }

    # 设置所有的PVPanel实例的初值(空字典{})
    all_pv_dict = {}.fromkeys(MANUFACTURER)
    all_pv_list = []

    # 为所有的PVPanel实例赋值
    for this_manufacturer in MANUFACTURER:
        all_pv_dict[this_manufacturer] = {}
        for this_configuration in CONFIGURATION:
            col_idx = [x for x in raw_data.columns if re.search(this_manufacturer, x, re.I)]
            col_idx = [x for x in col_idx if re.search(this_configuration, x, re.I)]
            if not col_idx:
                all_pv_dict[this_manufacturer][this_configuration] = None
                continue
            else:
                power_column = [x for x in col_idx if re.search('P_', x, re.I)][0]
                temperature_column = [x for x in col_idx if re.search('T_', x, re.I)][0]
                measurements = copy.deepcopy(raw_data[common_columns + [power_column] + [temperature_column]])
                measurements.rename(
                    mapper=dict(ChainMap(rename_mapper, {
                        power_column: 'power output',
                        temperature_column: 'panel temperature'
                    })),
                    axis=1,
                    inplace=True
                )
                this_pv_panel_obj = PVPanel(
                    measurements, manufacturer=this_manufacturer,
                    configuration=this_configuration,
                    obj_name='obj_name',
                    predictor_names=('tracker irradiation' if this_configuration == 'tracker' else 'fixed irradiation',
                                     'environmental temperature'),
                    dependant_names=('power output',)
                )
                if try_to_filer_using_2019_results:
                    outlier_mask = load_npy_file(this_pv_panel_obj.default_results_saving_path['outlier'])
                    this_pv_panel_obj[outlier_mask != 0] = np.nan
                all_pv_dict[this_manufacturer][this_configuration] = this_pv_panel_obj
                all_pv_list.append(this_pv_panel_obj)

    return all_pv_dict, all_pv_list


def check_data_availability_from_raw_data():
    # %% 载入原始数据
    raw_data = load_raw_data()
    time_data = raw_data['Time'].values
    hist(time_data, bins=9 * 12, x_label='Year', y_label='Frequency')


if __name__ == '__main__':
    check_data_availability_from_raw_data()
