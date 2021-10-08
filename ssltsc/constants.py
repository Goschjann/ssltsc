"""
constants module

stores constants used for model, architectures, trainers etc.
"""

#####################
# Feature Extractor

FC_PARAMETERS_1 = {
    'abs_energy': None,
    'absolute_sum_of_changes': None,
    'count_above_mean': None,
    'count_below_mean': None,
    'first_location_of_maximum': None,
    'first_location_of_minimum': None,
    'has_duplicate': None,
    'has_duplicate_max': None,
    'has_duplicate_min': None,
    'kurtosis': None,
    'last_location_of_maximum': None,
    'last_location_of_minimum': None,
    'length': None,
    'longest_strike_above_mean': None,
    'longest_strike_below_mean': None,
    'maximum': None,
    'minimum': None,
    'mean': None,
    'mean_abs_change': None,
    'mean_change': None,
    'mean_second_derivative_central': None,
    'median': None,
}

FC_PARAMETERS_2 = {
    'abs_energy': None,
    'absolute_sum_of_changes': None,
    'count_above_mean': None,
    'count_below_mean': None,
    'first_location_of_maximum': None,
    'first_location_of_minimum': None,
    'has_duplicate': None,
    'has_duplicate_max': None,
    'has_duplicate_min': None,
    'kurtosis': None,
    'last_location_of_maximum': None,
    'last_location_of_minimum': None,
    'length': None,
    'longest_strike_above_mean': None,
    'longest_strike_below_mean': None,
    'maximum': None,
    'minimum': None,
    'mean': None,
    'mean_abs_change': None,
    'mean_change': None,
    'mean_second_derivative_central': None,
    'median': None,
    'sample_entropy': None,
    'skewness': None,
    'standard_deviation': None,
    'sum_of_reoccurring_data_points': None,
    'sum_of_reoccurring_values': None,
    'sum_values': None,
    'variance': None,
    'variance_larger_than_standard_deviation': None,
}

#####################
# Data Sets

MANUAL_DATASETS_DICT = {'pamap2': 'PAMAP2_Dataset/',
                        'wisdm': 'WISDM_ar_v1.1/',
                        'sits': 'SITS_Dataset/',
                        'cifar10': 'cifar10/',
                        'svhn': 'svhn/',
                        'crop': 'Crop/',
                        'fordb': 'FordB/',
                        'electricdevices': 'ElectricDevices/',
                        'simulated': 'simulated/',
                        'sits_balanced': 'SITS_Balanced_Dataset/',
                        'sits_hard_balanced': 'SITS_Hard_Balanced_Dataset/'}

TRAINDATA_SIZE_DICT = {'pamap2': 11451,
                       'wisdm': 54907,
                       'sits': 90000,
                       'crop': 7200,
                       'electricdevices': 8926,
                       'fordb': 3636,
                       'cifar10': 50000}

# COLOR_MODEL_DICT = {'randomforest': {'color': 'dodgerblue', 'linestyle': ':'},
#                     'logisticregression': {'color': 'lightseagreen', 'linestyle': ':'},
#                     'labelspreading': {'color': 'black', 'linestyle': ':'},
#                     'supervised': {'color': 'blue', 'linestyle': ':'},
#                     'meanteacher': {'color': 'sienna', 'linestyle': '-'},
#                     'mixmatch': {'color': 'purple', 'linestyle': '-'},
#                     'vat': {'color': 'orange', 'linestyle': '-'},
#                     'ladder': {'color': 'forestgreen', 'linestyle': '-'},}
# #800000
# 'selfsupervised': {'color': '#99B181', 'linestyle': '-', 'name': 'Self-Supervised'}}

# jco color palette
# https://cran.r-project.org/web/packages/ggsci/vignettes/ggsci.html#non-ggplot2-graphics
COLOR_MODEL_DICT = {'randomforest': {'color': '#003C67FF', 'linestyle': ':', 'name': 'Random Forest'},
                    'logisticregression': {'color': '#7AA6DCFF', 'linestyle': ':', 'name': 'Logistic Regression'},
                    #'labelpropagation': {'color': '#3B3B3BFF', 'linestyle': ':', 'name': 'Label Propagation'},
                    'supervised': {'color': '#0073C2FF', 'linestyle': ':', 'name': 'Supervised FCN'},
                    'fully_supervised': {'color': 'black', 'linestyle': '--', 'name': 'Fully Labelled FCN'},
                    'meanteacher': {'color': '#FFB2B2', 'linestyle': '-', 'name': 'Mean Teacher'},
                    'mixmatch': {'color': '#CD534CFF', 'linestyle': '-', 'name': 'MixMatch'},
                    'vat': {'color': '#EFC000FF', 'linestyle': '-', 'name': 'VAT'},
                    'ladder': {'color': '#8F7700FF', 'linestyle': '-', 'name': 'Ladder'},
                    'selfsupervised': {'color': '#6B8E23', 'linestyle': '-', 'name': 'Self-Supervised'}}

COLOR_MODEL_DICT_SLIDES = {'supervised': {'color': '#0073C2FF', 'linestyle': ':', 'name': 'Supervised'},
                           'fully_supervised': {'color': 'black', 'linestyle': '--', 'name': 'Fully Labelled'},
                           'mixmatch': {'color': '#CD534CFF', 'linestyle': '-', 'name': 'Semi-Supervised'}}

DATASET_NAMES_DICT = {'pamap2': 'Pamap2',
                      'wisdm': 'WISDM',
                      'sits': 'SITS',
                      'sits_hard_balanced': 'SITS',
                      'crop': 'Crop',
                      'fordb': 'FordB',
                      'electricdevices': 'Electric Devices'}

SUPERVISED_BASELINE = {'pamap2': 0.98,
                       'wisdm': 0.995,
                       'sits': 0.97,
                       'sits_hard_balanced':  0.97,
                       'crop': 0.97,
                       'fordb': 0.85,
                       'electricdevices': 0.93}