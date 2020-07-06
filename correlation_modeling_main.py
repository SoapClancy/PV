from initialise_pv_using_raw_data_and_then_filter_Func import initialise_pv_using_raw_data_and_then_filter
import numpy as np
from PVPanel_Class import PVPanel


def run_correlation_modeling_main():
    all_pv_panel = initialise_pv_using_raw_data_and_then_filter()
    manufacturer = ('Sanyo', 'STP', 'DF', 'Yingli')
    configuration = ('open', 'closed', 'tracker')

    for this_manufacturer in manufacturer:
        for this_configuration in configuration:
            this_pv = all_pv_panel[this_manufacturer][this_configuration]  # type: PVPanel
            if this_pv is None:
                continue
            this_pv.fit_joint_probability_model_by_gmcm(used_outlier_category_for_modelling=(0,))

    #
    # """
    # ·   除了标准的outlier外，在“power_output - panel_temperature - irradiation”的建模中，新增加了"low_irradiation_outlier"，
    # 因为太多数据集中在low_irradiation区域了，形成统计上的singularity，无法fit，而且本身这里也没有uncertainty。所以，
    # irradiation < 10 的记录被设为outlier，outlier_category设为100
    # """
    #         mask = this_pv.measurements['irradiation'].values < 10
    #         this_pv.add_new_outlier_category(mask, 100)
    #         print(this_pv)
    #         # this_pv.fit_conditional_probability_model_by_gmm(('irradiation',), used_outlier_category_for_modelling=(0,),
    #         #                                                  bin_step=10, write_fitting_results=True)
    #         print(this_pv)
    #         this_pv.fit_conditional_probability_model_by_gmm(('panel temperature',),
    #                                                          used_outlier_category_for_modelling=(0,),
    #                                                          bin_step=2, write_fitting_results=True)


if __name__ == '__main__':
    all_pv_panel = initialise_pv_using_raw_data_and_then_filter()
    this_pv = all_pv_panel['Sanyo']['open']  # type: PVPanel

    # 条件概率模型的fit和应用
    # test_region = 10
    # x1 = this_pv.measurements['irradiation'].values[this_pv.outlier_category == 0][:test_region]
    # x2 = this_pv.measurements['panel temperature'].values[this_pv.outlier_category == 0][:test_region]
    # actual_y = this_pv.measurements['power output'].values[this_pv.outlier_category == 0][:test_region]
    #
    # this_pv.fit_conditional_probability_model_by_gmm(('irradiation', 'panel temperature'),
    #                                                  used_outlier_category_for_modelling=(0,),
    #                                                  bin_step=0.0005, write_fitting_results=False)
    #
    # 联合概率模型的fit
    # this_pv.fit_joint_probability_model_by_ecopula(used_outlier_category_for_modelling=(0,))

    test_region = 20
    mask = np.bitwise_and(this_pv.outlier_category == 0,
                          this_pv.measurements['power output'].values > 0)
    x1 = this_pv.measurements['irradiation'].values[mask][:test_region]
    x2 = this_pv.measurements['panel temperature'].values[mask][:test_region]
    actual_y = this_pv.measurements['power output'].values[mask][:test_region]

    #
    irradiation_model = this_pv.cal_power_output_using_conditional_probability_model_by_gmm(
        ('irradiation',),
        used_outlier_category_for_modelling=(0,),
        bin_step=10,
        predictor_var_value_in_tuple=(x1,))
    irradiation_model_mean = np.array(list(map(lambda x: x.mean_, irradiation_model)))

    #
    panel_temperature_model = this_pv.cal_power_output_using_conditional_probability_model_by_gmm(
        ('panel temperature',),
        used_outlier_category_for_modelling=(0,),
        bin_step=2,
        predictor_var_value_in_tuple=(x2,))
    panel_temperature_model_mean = np.array(list(map(lambda x: x.mean_, panel_temperature_model)))

    # 检验联合概率模型
    gmcm_model = this_pv.cal_power_output_using_joint_probability_model_by_gmcm(
        predictor_var_name_in_tuple=('panel temperature', 'irradiation'),
        used_outlier_category_for_modelling=(0,),
        predictor_var_value_in_tuple=(x2, x1))
    gmcm_mean = np.array([int(x.mean_) for x in gmcm_model])

    compare = np.stack((actual_y, irradiation_model_mean, gmcm_mean, panel_temperature_model_mean), axis=1)

    irradiation_model_error = np.abs(irradiation_model_mean - actual_y)
    print('irradiation_model_error mean={}'.format(np.mean(irradiation_model_error)))

    gmcm_model_error = np.abs(gmcm_mean - actual_y)
    print('gmcm_model_error mean={}'.format(np.mean(gmcm_model_error)))

    for i in range(test_region):
        gca = irradiation_model[i].plot_pdf(np.arange(0, 1250), label='irradiation only')
        # panel_temperature_model[i].plot_pdf(np.arange(0, 1250), ax=gca,label='panel temperature')
        gmcm_model[i].plot_pdf_like_ndarray(gca, label='GMCM', c='b', linestyle=':', x_label='Power output (W)',
                                            y_label='PDF', x_lim=(5, 1250))
