import pandas as pd
import numpy as np
from numpy import ndarray
import Ploting
from Ploting.fast_plot_Func import scatter, hist
from matplotlib import pyplot as plt
from project_path_Var import project_path_
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple, np_datetime64_to_datetime
from Writting.collections_Func import put_list_png_file_into_a_docx, put_all_png_in_a_path_into_a_docx
import os
import re
import copy
from pandas.plotting import register_matplotlib_converters
from Time_Processing.SynchronousTimeSeriesData_Class import SynchronousTimeSeriesData
from File_Management.load_save_Func import load_exist_npy_file_otherwise_run_and_save, \
    load_exist_pkl_file_otherwise_run_and_save, save_npy_file, load_pkl_file, save_pkl_file
from File_Management.path_and_file_management_Func import try_to_find_file, try_to_find_path_otherwise_make_one
from typing import Union, Tuple
from BivariateAnalysis_Class import Bivariate, BivariateOutlier, MethodOfBins
from Filtering.sklearn_novelty_and_outlier_detection_Func import use_isolation_forest
from UnivariateAnalysis_Class import CategoryUnivariate, UnivariateGaussianMixtureModel, UnivariatePDFOrCDFLike
from Correlation_Modeling.Copula_Class import GMCM, EmpiricalCopulaEx, EmpiricalCopula
from HighDimensionalAnalysis_Class import HighDimensionalAnalysis

register_matplotlib_converters()


class PVPanel:
    __slots__ = ('manufacturer', 'configuration', 'measurements', 'outlier_category')
    results_path = project_path_ + 'Data/Results/'
    time_resolution = 5

    """
    PV大类，包含各种方法
    manufacturer: PV的名字/厂商
    configuration：open 或 closed 或 Tracker
    measurements: 一个pd.DataFrame，包含同步测量的数据，DataFrame.columns包含:
        'ID',
        'time',
        'rank in a day',
        'power output',
        'panel temperature',
        'environmental temperature',
        'fixed irradiation',
        'tracker irradiation',
        'irradiation',→这个就是'fixed irradiation'或者'tracker irradiation'
        'wind speed'
    outlier_category: 异常值分类
    """

    def __init__(self, *, manufacturer: str, configuration: str, measurements: pd.DataFrame,
                 outlier_category: ndarray = None):
        self.manufacturer = manufacturer  # type: str
        self.configuration = configuration  # type: str
        self.measurements = measurements  # type:pd.DataFrame
        if configuration == 'tracker':
            self.measurements['irradiation'] = self.measurements['tracker irradiation']
        else:
            self.measurements['irradiation'] = self.measurements['fixed irradiation']
        self.outlier_category = outlier_category or self.identify_outlier()

    def __str__(self):
        t1 = np_datetime64_to_datetime(self.measurements['time'].values[0]).strftime('%Y-%m-%d %H.%M')
        t2 = np_datetime64_to_datetime(self.measurements['time'].values[-1]).strftime('%Y-%m-%d %H.%M')
        return "{} PV panel in {} configuration from {} to {}".format(self.manufacturer,
                                                                      self.configuration,
                                                                      t1,
                                                                      t2)

    def fast_plot_and_write(self, show_category_as_in_outlier: Union[int, tuple], **kwargs):
        """
        快速画变量关系图并写报告
        """
        self.fast_plot_bivariate_scatter(show_category_as_in_outlier, **kwargs)
        # self.fast_plot_time_rank_to_power()
        # 生成报告
        files = os.listdir(self.results_path)
        files = [x for x in files if re.search(self.manufacturer + ' ' + self.configuration, x)]
        files = [self.results_path + x for x in files]
        files = sorted(files, key=lambda x: os.path.getctime(x))
        put_list_png_file_into_a_docx(files, self.results_path + self.manufacturer + ' ' + self.configuration + '.docx')

    @classmethod
    def cal_recording_rank_in_a_day(cls, date_and_time: pd.Series) -> pd.DataFrame:
        date_and_time = copy.deepcopy(date_and_time.values)
        rank_in_a_day = [int(x.hour * 12 + int(x.minute / cls.time_resolution))
                         for x in datetime64_ndarray_to_datetime_tuple(date_and_time)]
        return pd.DataFrame({'rank_in_a_day': rank_in_a_day})

    def __prepare_fitting_conditional_probability_model(self, predictor_var_name_in_tuple: Tuple[str, ...],
                                                        used_outlier_category_for_modelling: Tuple[
                                                            int, ...],
                                                        path_: str):
        mask = CategoryUnivariate(self.outlier_category).cal_tuple_category_mask(
            used_outlier_category_for_modelling)
        dependent_var = self.measurements['power output'].values[mask]
        dependent_var_name = 'Power output (W)'
        if predictor_var_name_in_tuple.__len__() > 1:
            # 用GMCM降维
            ndarray_data = np.array([self.measurements[predictor_var_name_in_tuple[0]].values[mask],
                                     self.measurements[predictor_var_name_in_tuple[1]].values[mask]]).T
            gmcm = GMCM(ndarray_data=ndarray_data,
                        gmcm_model_file_=path_ + 'gmcm_model.mat',
                        marginal_distribution_file_=path_ + 'gmcm_marginal.pkl',
                        gmcm_fitting_k=10,
                        gmcm_max_fitting_iteration=1500)
            predictor_var = gmcm.cal_copula_cdf(ndarray_data_like=ndarray_data)
            predictor_var_name = 'Copula CDF'
            pass
        else:
            predictor_var = self.measurements[predictor_var_name_in_tuple[0]].values[mask]
            if predictor_var_name_in_tuple[0] == 'irradiation':
                predictor_var_name = 'Irradiation (W/$\mathregular{m}^2$)'
            elif predictor_var_name_in_tuple[0] == 'panel temperature':
                predictor_var_name = 'Panel temperature ($^\circ$C)'
            else:
                raise Exception("'predictor_var_name' is wrong")

        return predictor_var, dependent_var, predictor_var_name, dependent_var_name

    def fit_conditional_probability_model_by_gmm(self, predictor_var_name_in_tuple: Tuple[str, ...], *, bin_step: float,
                                                 used_outlier_category_for_modelling: Tuple,
                                                 write_fitting_results=False, **kwargs) -> dict:
        path_ = self.results_path + 'conditional_probability_by_gmm/' + self.__str__() + '/predictor=' + str(
            predictor_var_name_in_tuple) + ' category=' + str(
            used_outlier_category_for_modelling) + ' bin_step={}/'.format(bin_step)
        try_to_find_path_otherwise_make_one(path_)
        # 准备fitting data
        predictor_var, dependent_var, predictor_var_name, dependent_var_name = \
            self.__prepare_fitting_conditional_probability_model(predictor_var_name_in_tuple,
                                                                 used_outlier_category_for_modelling,
                                                                 path_)
        bivariate = Bivariate(predictor_var, dependent_var,
                              predictor_var_name=predictor_var_name,
                              dependent_var_name=dependent_var_name, bin_step=bin_step)

        @load_exist_pkl_file_otherwise_run_and_save(path_ + 'model.pkl')
        def load_or_make():
            return bivariate.fit_mob_using_gaussian_mixture_model(**kwargs)

        fitting_results = load_or_make  # type: dict
        if write_fitting_results:
            # 画fitting的目标数据(i.e., 原始数据)
            bivariate.plot_scatter(title=self.manufacturer + ' ' + self.configuration, label='Measurements',
                                   save_file_=path_ + 'original_data')
            plt.close()
            bivariate.plot_mob_uncertainty(title=self.manufacturer + ' ' + self.configuration,
                                           save_file_=path_ + 'original_data_uncertainty')
            plt.close()
            # 画模型的各个输出
            for key, this_bin in fitting_results.items():
                if not this_bin['this_bin_is_empty']:
                    this_bin_model = UnivariateGaussianMixtureModel(this_bin['this_bin_probability_model'],
                                                                    univariate_data=bivariate.mob[key][
                                                                        'dependent_var_in_this_bin'])
                    common_title = '{} bin boundary=[{}, {})'.format(predictor_var_name,
                                                                     this_bin['this_bin_boundary'][0],
                                                                     this_bin['this_bin_boundary'][-1])
                    this_bin_model.plot_pdf(x=np.arange(-1, 1255, 1), show_hist=True,
                                            title=common_title + ' pdf', x_label=dependent_var_name,
                                            label='GMM fitting PDF',
                                            save_file_=path_ + 'bin = {} pdf'.format(this_bin['this_bin_boundary']))
                    plt.close()
                    this_bin_model.plot_cdf(x=np.arange(-1, 1255, 1), show_ecdf=True,
                                            title=common_title + ' cdf', x_label=dependent_var_name,
                                            label='GMM fitting CDF',
                                            save_file_=path_ + 'bin = {} cdf'.format(this_bin['this_bin_boundary']))
                    plt.close()
            # 写入docx
            put_all_png_in_a_path_into_a_docx(path_, path_ + self.__str__() + '.docx')

        return fitting_results

    def cal_power_output_using_conditional_probability_model_by_gmm(self,
                                                                    predictor_var_name_in_tuple: Tuple[str, ...],
                                                                    *, bin_step: float,
                                                                    used_outlier_category_for_modelling: Tuple[
                                                                        int, ...],
                                                                    predictor_var_value_in_tuple: Tuple[ndarray, ...],
                                                                    if_no_available_mode: Union[int, str] =
                                                                    'nearest_not_none_bin_keys') \
            -> Tuple[UnivariateGaussianMixtureModel, ...]:
        """
        通过给定的predictor_var_value_in_tuple去估计power_output
        :param predictor_var_name_in_tuple: 声明模型的predictor_var_name，主要用于选择model
        :param bin_step:声明模型的bin_step，主要用于选择model
        :param used_outlier_category_for_modelling:声明模型的用于模型fitting的outlier category，主要用于选择model
        :param predictor_var_value_in_tuple:模型的真实输入，是predictor_var_name对应的数组
        :param if_no_available_mode: 如果(基于历史数据的bin)没有有效的模型怎么办。可选办法：
        1) 赋值‘nearest_not_none_bin_keys’， 选择最近的bin的有效模型
        2) 赋值一个int，选择n个最近的的bin的有效模型并组成一个list
        :return:UnivariateProbabilisticModel(或者它的subclass)组成的tuple，顺序对应predictor_var_value_in_tuple
        """
        path_ = self.results_path + 'conditional_probability_by_gmm/' + self.__str__() + '/predictor=' + str(
            predictor_var_name_in_tuple) + ' category=' + str(
            used_outlier_category_for_modelling) + ' bin_step={}/'.format(bin_step)
        # 载入或者fit模型，返回值是mob_fitting_like_dict
        model = load_pkl_file(path_ + 'model.pkl')
        if model is None:
            raise Exception("There is no model associated with the input parameters, please fit as first")
        # 生成predictor_var
        if predictor_var_name_in_tuple.__len__() > 1:
            # 用GMCM降维
            ndarray_data = np.array([predictor_var_value_in_tuple[0],
                                     predictor_var_value_in_tuple[1]]).T
            gmcm = GMCM(ndarray_data=ndarray_data,
                        gmcm_model_file_=path_ + 'gmcm_model.mat',
                        marginal_distribution_file_=path_ + 'gmcm_marginal.pkl')
            predictor_var = gmcm.cal_copula_cdf(ndarray_data_like=ndarray_data)
        else:
            predictor_var = predictor_var_value_in_tuple[0]
        # 计算(其实是选择)输出的条件概率模型
        power_output_model = []
        for this_predictor_var in predictor_var:
            this_model_idx = MethodOfBins.find_mob_key_according_to_mob_or_mob_fitting_like_dict(this_predictor_var,
                                                                                                 model)
            if if_no_available_mode == 'nearest_not_none_bin_keys':
                power_output_model.append(
                    UnivariateGaussianMixtureModel(model[this_model_idx['nearest_not_none_bin_keys']]
                                                   ['this_bin_probability_model']))
            else:
                assert isinstance(if_no_available_mode, int)
                temp = []
                for i in range(if_no_available_mode):
                    temp = UnivariateGaussianMixtureModel(model[this_model_idx['not_none_bin_keys'][i]]
                                                          ['this_bin_probability_model'])
                power_output_model.append(temp)
        return tuple(power_output_model)

    def __prepare_fitting_joint_probability_model(self, var_name_in_tuple: Tuple[str, ...],
                                                  used_outlier_category_for_modelling: Tuple):
        mask = CategoryUnivariate(self.outlier_category).cal_tuple_category_mask(
            used_outlier_category_for_modelling)
        data_to_be_fitted = np.full((int(np.sum(mask)), var_name_in_tuple.__len__()), np.nan)
        for this_col_idx, this_dimension in enumerate(var_name_in_tuple):
            data_to_be_fitted[:, this_col_idx] = self.measurements[this_dimension].values[mask]
        return data_to_be_fitted

    def fit_joint_probability_model_by_gmcm(self,
                                            var_name_in_tuple=(
                                                    'power output', 'irradiation', 'panel temperature'), *,
                                            used_outlier_category_for_modelling: Tuple[int, ...]):
        """
        拟合一个GMCM联合概率模型
        :param var_name_in_tuple: 声明联合概率模型的var_name，通过self.__prepare_fitting_joint_probability_model()方法
        查找到var_name对应的measurements，默认第一维是'power output'
        :param used_outlier_category_for_modelling: 选择用于模型拟合的outlier
        :return: None。但是GMCM模型会被存到path_+下
        """
        path_ = self.results_path + 'joint_probability_model_by_gmcm/' + self.__str__() + '/var=' + str(
            var_name_in_tuple) + ' category=' + str(
            used_outlier_category_for_modelling) + '/'
        try_to_find_path_otherwise_make_one(path_)

        # 准备fitting data
        data_to_be_fitted = self.__prepare_fitting_joint_probability_model(var_name_in_tuple,
                                                                           used_outlier_category_for_modelling)
        GMCM(ndarray_data=data_to_be_fitted,
             gmcm_model_file_=path_ + 'gmcm_model.mat',
             marginal_distribution_file_=path_ + 'marginal.pkl',
             gmcm_fitting_k=20,
             gmcm_max_fitting_iteration=2500)

    def fit_joint_probability_model_by_ecopula(self,
                                               var_name_in_tuple=(
                                                       'power output', 'irradiation', 'panel temperature'), *,
                                               used_outlier_category_for_modelling: Tuple[int, ...]):
        """
        拟合一个经验Copula联合概率模型
        :param var_name_in_tuple: 声明联合概率模型的var_name，通过self.__prepare_fitting_joint_probability_model()方法
        查找到var_name对应的measurements，默认第一维是'power output'
        :param used_outlier_category_for_modelling: 选择用于模型拟合的outlier
        :return: EmpiricalCopula
        """
        path_ = self.results_path + 'joint_probability_model_by_ecopula/' + self.__str__() + '/var=' + str(
            var_name_in_tuple) + ' category=' + str(
            used_outlier_category_for_modelling) + '/'
        try_to_find_path_otherwise_make_one(path_)
        # 准备fitting data
        data_to_be_fitted = self.__prepare_fitting_joint_probability_model(var_name_in_tuple,
                                                                           used_outlier_category_for_modelling)

        @load_exist_pkl_file_otherwise_run_and_save(path_ + 'model.pkl')
        def load_or_make():
            return EmpiricalCopulaEx(ndarray_data=data_to_be_fitted,
                                     marginal_distribution_file_=path_ + 'marginal.pkl').ecopula

        return load_or_make

    @staticmethod
    def __prepare_calculating_joint_probability_model(var_name_in_tuple, predictor_var_name_in_tuple,
                                                      predictor_var_value_in_tuple):
        # 将predictor_var_name_in_tuple转成conditional_var_idx，
        # 并且将predictor_var_value_in_tuple转成标准的ndarray_like的形式
        conditional_var_idx = []
        ndarray_data_like = np.full((predictor_var_value_in_tuple[0].shape[0],
                                     var_name_in_tuple.__len__()), np.nan)
        for this_predictor_var_name in predictor_var_name_in_tuple:
            for i, this_var_name in enumerate(var_name_in_tuple):
                if this_predictor_var_name == this_var_name:
                    conditional_var_idx.append(i)  # i表示predictor_var_value_in_tuple中的元素对应var_name_in_tuple的位置
                    ndarray_data_like[:, i] = predictor_var_value_in_tuple[
                        predictor_var_name_in_tuple.index(this_predictor_var_name)]
        return ndarray_data_like, conditional_var_idx

    def cal_power_output_using_joint_probability_model_by_gmcm(self,
                                                               var_name_in_tuple=(
                                                                       'power output', 'irradiation',
                                                                       'panel temperature'),
                                                               *, predictor_var_name_in_tuple: Tuple[str, ...],
                                                               used_outlier_category_for_modelling: Tuple[int, ...],
                                                               predictor_var_value_in_tuple: Tuple[ndarray, ...]) \
            -> Tuple[UnivariatePDFOrCDFLike, ...]:
        path_ = self.results_path + 'joint_probability_model_by_gmcm/' + self.__str__() + '/var=' + str(
            var_name_in_tuple) + ' category=' + str(
            used_outlier_category_for_modelling) + '/'
        ndarray_data_like, conditional_var_idx = \
            self.__prepare_calculating_joint_probability_model(var_name_in_tuple,
                                                               predictor_var_name_in_tuple,
                                                               predictor_var_value_in_tuple)
        # 计算gmcm_conditional_cdf
        model = GMCM(gmcm_model_file_=path_ + 'gmcm_model.mat',
                     marginal_distribution_file_=path_ + 'marginal.pkl')
        gmcm_conditional_cdf = model.cal_copula_conditional_cdf(ndarray_data_like,
                                                                conditional_var_idx=tuple(conditional_var_idx))
        # 将gmcm_conditional_cdf转成UnivariateCDFLike类以便分析(采样，算mean_，算inverse_cdf)
        return tuple(UnivariatePDFOrCDFLike(x) for x in gmcm_conditional_cdf)

    def cal_power_output_using_joint_probability_model_by_ecopula(self,
                                                                  var_name_in_tuple=(
                                                                          'power output', 'irradiation',
                                                                          'panel temperature'),
                                                                  *, predictor_var_name_in_tuple: Tuple[str, ...],
                                                                  used_outlier_category_for_modelling: Tuple[int, ...],
                                                                  predictor_var_value_in_tuple: Tuple[ndarray, ...]) \
            -> Tuple[UnivariatePDFOrCDFLike, ...]:
        path_ = self.results_path + 'joint_probability_model_by_ecopula/' + self.__str__() + '/var=' + str(
            var_name_in_tuple) + ' category=' + str(
            used_outlier_category_for_modelling) + '/'
        ndarray_data_like, conditional_var_idx = \
            self.__prepare_calculating_joint_probability_model(var_name_in_tuple,
                                                               predictor_var_name_in_tuple,
                                                               predictor_var_value_in_tuple)
        ecopula = load_pkl_file(path_ + 'model.pkl')
        if ecopula is None:
            raise Exception("There is no model associated with the input parameters, please fit as first")
        # 计算gmcm_conditional_cdf
        model = EmpiricalCopulaEx(marginal_distribution_file_=path_ + 'marginal.pkl',
                                  ecopula=ecopula)

        ecopula_conditional_cdf = model.cal_copula_conditional_cdf(ndarray_data_like,
                                                                   conditional_var_idx=tuple(conditional_var_idx))
        # 将gmcm_conditional_cdf转成UnivariateCDFLike类以便分析(采样，算mean_，算inverse_cdf)
        return tuple(UnivariatePDFOrCDFLike(x) for x in ecopula_conditional_cdf)

    def fast_plot_bivariate_scatter(self, show_category_as_in_outlier: Union[Tuple[int, ...], str], **kwargs):
        """
        快速展示这个PV的measurements之间变量的一些关系，包括：
        irradiation与power output的scatter
        panel temperature与power output的scatter
        """
        title = self.manufacturer + ' ' + self.configuration
        common_save_path_and_name = self.results_path + self.manufacturer + ' ' + self.configuration
        # irradiation与power output的scatter
        bivariate = Bivariate(self.measurements['irradiation'].values, self.measurements['power output'].values,
                              predictor_var_name='Irradiation (W/$\mathregular{m}^2$)',
                              dependent_var_name='Power output (W)',
                              category=self.outlier_category)
        bivariate.plot_scatter(show_category=show_category_as_in_outlier,
                               title=title, save_format='png', alpha=0.75,
                               save_file_=common_save_path_and_name + '_irr_and_Pout' + str(
                                   show_category_as_in_outlier), **kwargs)
        # panel temperature与power output的scatter
        bivariate = Bivariate(self.measurements['panel temperature'].values, self.measurements['power output'].values,
                              predictor_var_name='Panel temperature ($^\circ$C)',
                              dependent_var_name='Power output (W)',
                              category=self.outlier_category)
        bivariate.plot_scatter(show_category=show_category_as_in_outlier,
                               title=title, save_format='png', alpha=0.75,
                               save_file_=common_save_path_and_name + '_pT_and_Pout' + str(
                                   show_category_as_in_outlier), **kwargs)

    def fast_plot_time_rank_to_power(self):
        """
        快速展示这个PV的measurements的power output与一天中的时间之间的关系
        """
        power = []
        for this_time_rank in range(0, 288):
            idx = self.measurements['rank_in_a_day'] == this_time_rank
            power.append(self.measurements['power output'].values[idx])

        @Ploting.fast_plot_Func.show_fig
        def wrapper():
            @Ploting.fast_plot_Func.creat_fig((5, 5 * 0.618))
            def fast_plot():
                for time, this_power in enumerate(power):
                    plt.scatter(np.full_like(this_power, time), this_power, c='b', s=2, rasterized=True)

        wrapper(x_label='Time in a day', y_label='Power output (W)',
                x_ticks=(range(0, 288, 72), ('0:00', '6:00', '12:00', '18:00')),
                x_lim=(0, 288),
                title=self.manufacturer + ' ' + self.configuration,
                save_file_=self.results_path + self.manufacturer + ' ' + self.configuration + '_6',
                save_format='png')

    def identify_outlier(self) -> ndarray:
        @load_exist_npy_file_otherwise_run_and_save(
            self.results_path + 'Filtering/' + self.__str__() + ' identify_outlier_results.npy')
        def load_or_make():
            self.outlier_category = np.full(self.measurements.shape[0], 0)
            self.outlier_category[self.__identify_missing_data_outlier()] = -1
            self.outlier_category[self.__identify_shut_down_outlier()] = 1
            self.outlier_category[self.__identify_change_point_outlier()] = 2
            self.outlier_category[self.__identify_linear_series_outlier()] = 3
            # self.outlier_category[self.__identify_interquartile_outlier()] = 4
            self.outlier_category[self.__identify_isolation_forest_outlier()] = 5
            return self.outlier_category

        return load_or_make

    def update_outlier_category(self, old_category: int, new_category: int):
        CategoryUnivariate(self.outlier_category).update_category(old_category, new_category)
        save_npy_file(self.results_path + 'Filtering/' + self.__str__() + ' identify_outlier_results.npy',
                      self.outlier_category)

    def add_new_outlier_category(self, new_outlier_category_mask: ndarray, new_outlier_assigned_number: int):
        """
        可能以后的分析中需要新添加outlier type，以满足具体的分析需要
        ·   除了标准的outlier外，在“power_output-panel_temperature-irradiation”的建模中，新增加了"low_irradiation_outlier"，
            因为太多数据集中在low_irradiation区域了，形成统计上的singularity，无法fit，而且本身这里也没有uncertainty。所以，
            irradiation<10的记录被设为outlier，outlier_category设为100
        """
        self.outlier_category[new_outlier_category_mask] = new_outlier_assigned_number
        save_npy_file(self.results_path + 'Filtering/' + self.__str__() + ' identify_outlier_results.npy',
                      self.outlier_category)

    def __identify_missing_data_outlier(self):
        """
        missing data outlier
        """
        synchronous_time_series_data = SynchronousTimeSeriesData(self.measurements, self.outlier_category)
        return synchronous_time_series_data.missing_data_outlier_in_tuple_synchronous_data(
            ('panel temperature', 'irradiation', 'power output'))

    def __identify_shut_down_outlier(self):
        """
        shut down outlier
        """
        considered_data_idx = self.outlier_category == 0
        bivariate_outlier = BivariateOutlier(predictor_var=self.measurements['irradiation'].values,
                                             dependent_var=self.measurements['power output'].values,
                                             mob_and_outliers_detection_considered_data_idx=considered_data_idx)
        return bivariate_outlier.identify_shut_down_outlier(cannot_be_zero_predictor_var_range=(10, np.inf),
                                                            zero_upper_tolerance_factor=0.1)

    def __identify_change_point_outlier(self):
        """
        change point outlier in series data
        """
        synchronous_time_series_data = SynchronousTimeSeriesData(self.measurements, self.outlier_category)
        return synchronous_time_series_data.change_point_outliers_in_tuple_synchronous_data(
            ('environmental temperature', 'panel temperature'), 6)

    def __identify_linear_series_outlier(self):
        """
        linear series outlier in series data
        """
        synchronous_time_series_data = SynchronousTimeSeriesData(self.measurements, self.outlier_category)
        return synchronous_time_series_data.linear_series_outliers_in_tuple_synchronous_data(
            ('environmental temperature', 'panel temperature'), 6)

    def __identify_interquartile_outlier(self):
        """
        bivariate interquartile outliers
        """
        #
        considered_data_mask = self.outlier_category == 0
        bivariate_outlier = BivariateOutlier(predictor_var=self.measurements['irradiation'].values,
                                             dependent_var=self.measurements['power output'].values,
                                             bin_step=10,
                                             mob_and_outliers_detection_considered_data_mask=considered_data_mask)
        return bivariate_outlier.identify_interquartile_outliers_based_on_method_of_bins()

    def __identify_isolation_forest_outlier(self):
        outlier = np.full(self.measurements.shape[0], False)
        # #
        # cannot_be_outlier_rule = (((1108, 1131), (996, 1090)),
        #                           ((1060, 1111), (998, 1072)),
        #                           ((1035, 1060), (998, 1061)),
        #                           ((1200, np.inf), (1000, np.inf)),
        #                           ((1019, 1065), (950, 1000)),
        #                           ((1069, 1147), (950, 1000)),
        #                           ((1040, np.inf), (1000, 1115)))
        # must_be_outlier_rule = (((0, 160), (270, np.inf)),
        #                         ((80, 125), (220, np.inf)),
        #                         ((210, np.inf), (20, 85)),
        #                         ((345, np.inf), (20, 200)),
        #                         ((680, np.inf), (20, 420)),
        #                         ((-np.inf, np.inf), (1115, np.inf)),
        #                         ((1010, np.inf), (20, 1000)),
        #                         ((1080, np.inf), (20, 1023)),
        #                         ((130, np.inf), (20, 65)),
        #                         ((100, np.inf), (20, 50)),
        #                         ((178, np.inf), (20, 110)))
        #
        # considered_data_idx = self.outlier_category == 0
        # bivariate = Bivariate(predictor_var=self.measurements['irradiation'].values,
        #                       dependent_var=self.measurements['power output'].values,
        #                       mob_and_outliers_detection_considered_data_idx=considered_data_idx,
        #                       cannot_be_outlier_rule=cannot_be_outlier_rule,
        #                       must_be_outlier_rule=must_be_outlier_rule)
        # isolation_forest = bivariate.identify_outlier_using_isolation_forest(
        #     isolationforest_args={'max_features': 2,
        #                           'max_samples': 1.0,
        #                           'contamination': 0.01,
        #                           'n_jobs': -1,
        #                           'verbose': 1})
        #
        # outlier = np.bitwise_or(outlier, isolation_forest)
        #
        # #
        # must_be_outlier_rule = (((-1, 1), (70, np.inf)),
        #                         ((47, np.inf), (-2, 2)))
        # considered_data_idx = self.outlier_category == 0
        # bivariate = Bivariate(predictor_var=self.measurements['panel temperature'].values,
        #                       dependent_var=self.measurements['power output'].values,
        #                       mob_and_outliers_detection_considered_data_idx=considered_data_idx,
        #                       must_be_outlier_rule=must_be_outlier_rule)
        # isolation_forest = bivariate.identify_outlier_using_isolation_forest(
        #     isolationforest_args={'max_features': 2,
        #                           'max_samples': 1.0,
        #                           'contamination': 0.0005,
        #                           'n_jobs': -1,
        #                           'verbose': 1})
        #
        # outlier = np.bitwise_or(outlier, isolation_forest)

        considered_data_idx = self.outlier_category == 0
        multi_dimension_data = np.array([self.measurements['power output'].values[considered_data_idx],
                                         self.measurements['panel temperature'].values[considered_data_idx],
                                         self.measurements['irradiation'].values[considered_data_idx]]).T
        data_idx = np.where(considered_data_idx)[0]
        temp = use_isolation_forest(multi_dimension_data, data_idx, {'n_estimators': 128,
                                                                     'max_features': 3,
                                                                     'max_samples': 1.0,
                                                                     'contamination': 0.0075,
                                                                     'n_jobs': -1,
                                                                     'verbose': 1})
        outlier[temp] = True

        return outlier
