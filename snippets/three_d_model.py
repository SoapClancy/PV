from prepare_datasets import initialise_pv_using_raw_data_and_then_filter
from File_Management.load_save_Func import *
from File_Management.path_and_file_management_Func import *
from PVPanel_Class import PVPanel
from Ploting.fast_plot_Func import *
from Correlation_Modeling.Copula_Class import THREE_DIM_CVINE_CONSTRUCTION

"""
This (remove_win10_max_path_limit) is a default call to make the code run on Windows platform 
(and only Windows 10 is supported!)
Because the Windows 10 default path length limitation (MAX_PATH) is 256 characters, many load/save functions in
this project may have errors in reading the path
To restore the path_limit, call File_Management.path_management_Func.restore_win10_max_path_limit yourself
"""
remove_win10_max_path_limit()

ALL_PV_PANELS = initialise_pv_using_raw_data_and_then_filter()[1]


def fit_3_d_model(this_pv_obj: PVPanel):
    normal_data_mask = load_npy_file(this_pv_obj.default_results_saving_path['outlier']) == 0
    this_pv_obj[normal_data_mask].fit_joint_probability_model_by_copula(THREE_DIM_CVINE_CONSTRUCTION)


def fit_3_d_model_slice_input(slice_input: slice):
    for this_pv_obj in ALL_PV_PANELS[slice_input]:
        fit_3_d_model(this_pv_obj)


if __name__ == '__main__':
    # %% To fit
    # for _this_pv_obj in ALL_PV_PANELS[slice(0, 5)]:
    #     fit_3_d_model(_this_pv_obj)

    # %% To estimate
    # Example:
    # solar irradiance = 500;
    # environmental temperature = 10
    _this_pv_obj = ALL_PV_PANELS[0]
    print(f"Model is from {_this_pv_obj}")
    pdf_tuple = _this_pv_obj.obtain_pdf_by_copula(
        THREE_DIM_CVINE_CONSTRUCTION,  # Define Vine structure
        np.array([[500, 10]]),  # solar irradiance = 500, environmental temperature = 10
        pdf_x_limits=[0, 1250]  # define a limit for pdf sampling, which should >= limit of power output
    )
    # Once obtain pdf_tuple, its elements can provide PDF, CDF, inverseCDF, etc.
    # e.g., PDF
    pdf_tuple[0].plot_pdf_like_ndarray(title='PDF')
    # e.g., CDF
    pdf_tuple[0].plot_cdf_like_ndarray(title='CDF')
    # e.g., inverseCDF
    series(np.arange(0, 1.0001, 0.0001),
           pdf_tuple[0].find_nearest_inverse_cdf(np.arange(0, 1.0001, 0.0001)),
           title='inverseCDF')
