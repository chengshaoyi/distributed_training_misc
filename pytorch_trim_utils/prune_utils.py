###################################
# Top level pruning functions     #
###################################
from collections import OrderedDict
import numpy as np

from pytorch_trim_utils.common_utils import generate_trimed_weight_at_axis
from pytorch_trim_utils.layer_pruning_utils import init_pruned_layer, \
    populate_pruned_layer

# filter level pruning
def prune_src_layer_filters(original_layer, weight_device, selector, **kwargs):
    filter_indices_to_remove = selector(original_layer, **kwargs)
    filter_removal_indices_cleaned = list(OrderedDict.fromkeys(filter_indices_to_remove))
    filter_removal_indices_ordered = sorted(filter_removal_indices_cleaned)
    new_layer = init_pruned_layer(original_layer, weight_device, filter_removal_indices_ordered)
    populate_pruned_layer(new_layer, original_layer, filter_removal_indices_ordered, weight_device)
    return new_layer, filter_removal_indices_ordered

def prune_dst_layer_filters(original_layer, weight_device, filter_removal_indices_ordered, original_src_layer):
    new_layer = init_pruned_layer(original_layer, weight_device, filter_removal_indices_ordered, original_src_layer)
    populate_pruned_layer(new_layer, original_layer, filter_removal_indices_ordered, weight_device, original_src_layer)
    return new_layer



