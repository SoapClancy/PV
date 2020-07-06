from initialise_pv_using_raw_data_and_then_filter_Func import initialise_pv_using_raw_data_and_then_filter

all_PVPanel = initialise_pv_using_raw_data_and_then_filter()

manufacturer = ('Sanyo', 'STP', 'DF', 'Yingli')
configuration = ('open', 'closed', 'tracker')

# for this_manufacturer in manufacturer:
#     for this_configuration in configuration:
#         this_pv = all_PVPanel[this_manufacturer][this_configuration]
#         if this_pv is None:
#             continue
#         # this_pv.fast_plot_bivariate_scatter('all')
#         # this_pv.fast_plot_bivariate_scatter((5, 3, 2, 1, 0), show_category_color=('r', 'b', 'aqua', 'k', 'g'),
#         #                                     show_category_label=('CAT-IV', 'CAT-III', 'CAT-II', 'CAT-I', 'Normal'))
#         this_pv.fast_plot_and_write((0,), show_category_color=('g',),
#                                     show_category_label=('Normal',))

this_pv = all_PVPanel['Sanyo']['open']
this_pv.fast_plot_bivariate_scatter((5, 3, 2, 1, 0), show_category_color=('r', 'b', 'aqua', 'k', 'g'),
                                    show_category_label=('CAT-IV', 'CAT-III', 'CAT-II', 'CAT-I', 'Normal'))
this_pv.fast_plot_bivariate_scatter((0,), show_category_color=('g',),
                                    show_category_label=('Normal',))
