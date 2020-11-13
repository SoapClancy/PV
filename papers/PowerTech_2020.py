from Ploting.fast_plot_Func import *
from prepare_datasets import initialise_pv_using_raw_data
from File_Management.load_save_Func import *
from File_Management.path_and_file_management_Func import *
import pandas as pd
import seaborn as sns
from itertools import product
from functools import reduce
from project_utils import *

one_pv_panel = initialise_pv_using_raw_data(try_to_filer_using_2019_results=True)[0]['DF']['tracker']
print(one_pv_panel)
# scatter(*one_pv_panel[['tracker irradiation', 'power output']].values.T)
WS_TEMP_PV_DATASET = one_pv_panel[['environmental temperature', 'wind speed']].pd_view()  # type: pd.DataFrame
# scatter(*TEMP_SOLRAD.values.T)
# g = sns.jointplot(data=TEMP_SOLRAD, x="environmental temperature", y="tracker irradiation")
# g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
# g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)

WS_TEMP_MERRA_DATASET = pd.read_csv(project_path_ / r"Data\Papers\PowerTech_2020\data.csv", skiprows=24)
# WS_TEMP_MERRA_DATASET = WS_TEMP_MERRA_DATASET[['Temperature', 'Wind speed']]
# WS_TEMP_MERRA_DATASET = WS_TEMP_MERRA_DATASET.rename(columns={
#     'Temperature': 'environmental temperature',
#     'Wind speed': 'wind speed'
# })
WS_TEMP_MERRA_DATASET = WS_TEMP_MERRA_DATASET[['Short-wave irradiation', 'Wind speed']]
WS_TEMP_MERRA_DATASET = WS_TEMP_MERRA_DATASET.rename(columns={
    'Short-wave irradiation': 'environmental temperature',
    'Wind speed': 'wind speed'
})

PCT_OUTER = [1, 2.5, 5, 95, 97.5, 99]
WS_ABS_SMALLER_THAN = [5, 10, ]
WS_ABS_LARGER_THAN = [15, 20, ]

###################################################################################################
_ws = WS_TEMP_MERRA_DATASET['wind speed'].values
_temperature = WS_TEMP_MERRA_DATASET['environmental temperature'].values
###################################################################################################
for mode in ('joint', 'WS|TEMP', 'TEMP|WS'):
    both_not_nan = ~np.bitwise_or(np.isnan(_ws), np.isnan(_temperature))
    _ws = _ws[both_not_nan]
    _temperature = _temperature[both_not_nan]
    data_size = _ws.shape[0]

    ws_pct_outer_abs_mixed = (PCT_OUTER +
                              list(map(lambda x: 100 * np.sum(_ws <= x) / data_size, WS_ABS_SMALLER_THAN)) +
                              list(map(lambda x: 100 * np.sum(_ws >= x) / data_size, WS_ABS_LARGER_THAN)))

    columns = pd.MultiIndex.from_product([['TEMP_PCT_OUTER'], PCT_OUTER], names=['temperature', 'percentile'])
    index = [product(['WS_PCT_OUTER'], PCT_OUTER),
             product(['WS_ABS_SMALLER_THAN'], WS_ABS_SMALLER_THAN),
             product(['WS_ABS_LARGER_THAN'], WS_ABS_LARGER_THAN)]
    index = reduce(lambda x, y: x + y, list(map(lambda z: list(z), index)))
    index = pd.MultiIndex.from_tuples(index, names=['ws', 'percentile or absolute'])
    results_df = pd.DataFrame(columns=columns, index=index, dtype=float)

    for i, this_ws_pct_outer in enumerate(ws_pct_outer_abs_mixed):  # Wind speed loop
        if this_ws_pct_outer < 50:
            this_condition_ws_pct_outer_mask = (_ws <= np.percentile(_ws, this_ws_pct_outer))
            this_condition_ws_pct_outer_mask_theoretical_sum = this_ws_pct_outer / 100
        else:
            this_condition_ws_pct_outer_mask = (_ws >= np.percentile(_ws, this_ws_pct_outer))
            this_condition_ws_pct_outer_mask_theoretical_sum = 1 - this_ws_pct_outer / 100
        # print(sum(this_condition_ws_pct_outer_mask))
        for j, this_temperature_pct_outer in enumerate(PCT_OUTER):
            if this_temperature_pct_outer < 50:
                this_condition_temperature_pct_outer_mask = \
                    (_temperature <= np.percentile(_temperature, this_temperature_pct_outer))
                this_condition_temperature_pct_outer_mask_theoretical_sum = this_temperature_pct_outer / 100
            else:
                this_condition_temperature_pct_outer_mask = \
                    (_temperature >= np.percentile(_temperature, this_temperature_pct_outer))
                this_condition_temperature_pct_outer_mask_theoretical_sum = 1 - this_temperature_pct_outer / 100

            joint_prob = np.sum(np.bitwise_and(this_condition_ws_pct_outer_mask,
                                               this_condition_temperature_pct_outer_mask)) / data_size
            if mode == 'joint':
                results_df.iloc[i, j] = joint_prob
            elif mode == 'WS|TEMP':
                results_df.iloc[i, j] = joint_prob / (np.sum(this_condition_temperature_pct_outer_mask) / data_size)
                # print(np.sum(this_condition_temperature_pct_outer_mask) / data_size)
            else:
                results_df.iloc[i, j] = joint_prob / (np.sum(this_condition_ws_pct_outer_mask) / data_size)
                # print(np.sum(this_condition_ws_pct_outer_mask) / data_size)

    results_df[:] *= 100
    if mode == 'joint':
        save_name = './joint.csv'
    elif mode == 'WS|TEMP':
        save_name = './WS_conditioned_on_TEMP.csv'
    else:
        save_name = './TEMP_conditioned_on_WS.csv'
    results_df.to_csv(save_name)

if __name__ == '__main__':
    pass
