import torch
from torch import nn as nn

from pytorch_trim_utils.common_utils import generate_trimed_weight_at_axis
from pytorch_trim_utils import GPU_TARGET, CPU_TARGET

LAYER_WEIGHT = 0
LAYER_BIAS = 1

def find_prune_multiplier(layer, src_layer=None):
    if src_layer is None:
        pass



# conv population
def populate_pruned_conv_coe_tensor(original_conv, new_conv, filter_removal_indices_ordered,
                                    weight_device, coe_type, src_layer=None):
    assert coe_type == LAYER_WEIGHT or coe_type == LAYER_BIAS, 'unrecognized coe type'
    original_conv_coe = get_coe_handle(original_conv, coe_type)
    new_conv_coe = get_coe_handle(new_conv, coe_type)
    if src_layer is None:
        original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_conv_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=0)
    elif coe_type == LAYER_WEIGHT:
        original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_conv_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=1)
    else:
        original_conv_coe_trimmed = original_conv_coe
    new_conv_coe[...] = original_conv_coe_trimmed
    commit_coe(new_conv, new_conv_coe, weight_device, coe_type)

def populate_pruned_linear_coe_tensor(original_linear, new_linear, filter_removal_indices_ordered,
                                      weight_device, coe_type, src_layer=None):
    assert coe_type == LAYER_WEIGHT or coe_type == LAYER_BIAS, 'unrecognized coe type'
    original_linear_coe = get_coe_handle(original_linear, coe_type)
    new_linear_coe = get_coe_handle(new_linear, coe_type)
    if src_layer is None:
        original_linear_coe_trimmed = generate_trimed_weight_at_axis(original_linear_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=0)
    elif coe_type == LAYER_WEIGHT:
        if isinstance(src_layer, nn.Conv2d):
            # here each removed index in the source translate to the
            #FIXME:
        elif isinstance(src_layer, nn.Linear):
            original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_linear_coe,
                                                                       filter_removal_indices_ordered,
                                                                       axis=1)
    else:
        original_linear_coe_trimmed = original_linear_coe
    new_linear_coe[...] = original_linear_coe_trimmed
    commit_coe(new_linear, new_linear_coe, weight_device, coe_type)


def init_pruned_layer(original_layer, weight_device, filter_indices_removal_ordered, src_layer=None):
    if isinstance(original_layer, nn.Conv2d):
        layer_init_fn = init_conv_layer
    elif isinstance(original_layer, nn.Linear):
        layer_init_fn = init_linear_layer
    return layer_init_fn(original_layer, weight_device, filter_indices_removal_ordered, src_layer)


def populate_pruned_layer(new_layer, original_layer, filter_removal_indices_ordered, weight_device, src_layer=None):

    populate_pruned_conv_coe_tensor(original_layer, new_layer, filter_removal_indices_ordered, weight_device,
                                    LAYER_WEIGHT, src_layer)
    if new_layer.bias is not None:
        populate_pruned_conv_coe_tensor(original_layer, new_layer, filter_removal_indices_ordered,
                                        weight_device, LAYER_BIAS, src_layer)




# Linear layer stuff
def init_linear_layer(original_linear, weight_device, filter_indices_removal_ordered, src_layer):
    num_filters_to_remove = len(filter_indices_removal_ordered)
    in_features = original_linear.in_features
    out_features = original_linear.out_features
    if src_layer is None:
        out_features -= num_filters_to_remove
    else:
        if isinstance(src_layer, nn.Linear):
            in_features -= num_filters_to_remove
        elif isinstance(src_layer, nn.Conv2d):
            to_reduce = num_filters_to_remove*in_features/src_layer.out_channels
            in_features -= to_reduce
    dup_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=original_linear.bias is not None)\
        .to(weight_device)
    return dup_linear


# CONV layer stuff
def init_conv_layer(original_conv, weight_device, filter_indices_removal_ordered, src_layer):
    num_filters_to_remove = len(filter_indices_removal_ordered)
    original_in_channel = original_conv.in_channels
    original_out_channel = original_conv.out_channels
    if src_layer is None:
        original_out_channel -= num_filters_to_remove
    else:
        assert isinstance(src_layer, nn.Conv2d)
        original_in_channel -= num_filters_to_remove
    dup_conv = nn.Conv2d(in_channels=original_in_channel, out_channels=original_out_channel,
                         kernel_size=original_conv.kernel_size, stride=original_conv.stride,
                         padding=original_conv.padding, dilation=original_conv.dilation, groups=original_conv.groups,
                         bias=original_conv.bias is not None) \
        .to(weight_device)
    return dup_conv




# end of conv stuff


def acquire_coe(layer, coe_type):
    assert coe_type == LAYER_WEIGHT or coe_type == LAYER_BIAS
    return layer.weight if coe_type == LAYER_WEIGHT else layer.bias

def get_coe_handle(conv, coe_type):
    return acquire_coe(conv, coe_type).data.cpu().numpy()

def commit_coe(conv, coefficients, target_device, coe_type):
    commit_fn = lambda coe : torch.from_numpy(coe).cuda() if target_device == GPU_TARGET else torch.from_numpy(coe)
    acquire_coe(conv, coe_type).data = commit_fn(coefficients)
