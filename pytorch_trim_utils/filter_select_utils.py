from pytorch_trim_utils.layer_pruning_utils import LAYER_WEIGHT, LAYER_BIAS, get_coe_handle
import numpy as np

def magnitude_based_filter_select(conv, **kwargs):
    # we find the filter with the largest sum of absolute value of the element in the weight
    filter_weight = get_coe_handle(conv, LAYER_WEIGHT)
    if conv.bias is not None:
        filter_bias = get_coe_handle(conv, LAYER_BIAS)
    else:
        filter_bias = np.zeros(filter_weight.shape[0])
    weight_sums = [np.sum(np.absolute(filter_weight[i, ...])) for i in range(filter_weight.shape[0])]
    bias_sums = np.absolute(filter_bias)
    total_magnitudes = weight_sums+bias_sums
    coe_ind = list(range(filter_weight.shape[0]))
    magnitude_dict = dict(zip(coe_ind, total_magnitudes))
    sorted_magnitude = sorted(magnitude_dict.items(), key=lambda x: x[1])
    #print(sorted_magnitude)
    num_of_filters_to_trim = kwargs['num']
    filter_to_trim = [x[0] for x in sorted_magnitude[:num_of_filters_to_trim]]
    print(filter_to_trim)
    return filter_to_trim