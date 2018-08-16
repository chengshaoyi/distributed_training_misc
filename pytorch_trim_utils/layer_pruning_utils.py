import torch
from torch import nn as nn

from pytorch_trim_utils.common_utils import generate_trimed_weight_at_axis
from pytorch_trim_utils import SRC_LAYER, DST_LAYER, GPU_TARGET, CPU_TARGET

LAYER_WEIGHT = 0
LAYER_BIAS = 1

def populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered,
                                    weight_device, layer_role,
                                    coe_type):
    assert layer_role == DST_LAYER or layer_role == SRC_LAYER, 'unrecognized trim dim'
    assert coe_type == LAYER_WEIGHT or coe_type == LAYER_BIAS, 'unrecognized coe type'
    original_conv_coe = get_coe_handle(src_conv, coe_type)
    new_conv_coe = get_coe_handle(new_src_conv, coe_type)
    if layer_role == SRC_LAYER:
        original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_conv_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=0)
    elif layer_role == DST_LAYER and coe_type == LAYER_WEIGHT:
        original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_conv_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=1)
    else:
        original_conv_coe_trimmed = original_conv_coe
    new_conv_coe[...] = original_conv_coe_trimmed
    commit_coe(new_src_conv, new_conv_coe, weight_device, coe_type)


def init_pruned_layer(layer, weight_device, filter_indices_removal_ordered, layer_role, src_layer=None):
    assert layer_role == DST_LAYER or layer_role == SRC_LAYER
    if isinstance(layer, nn.Conv2d):
        layer_init_fn = init_conv_layer
    elif isinstance(layer, nn.Linear):
        layer_init_fn = init_linear_layer
    return layer_init_fn(layer,weight_device,filter_indices_removal_ordered,layer_role, src_layer)


# Linear layer stuff
def init_linear_layer(original_linear, weight_device, filter_indices_removal_ordered, layer_role, src_layer):
    num_filters_to_remove = len(filter_indices_removal_ordered)
    assert layer_role == DST_LAYER or layer_role == SRC_LAYER
    in_features = original_linear.in_features
    out_features = original_linear.out_features
    if layer_role == SRC_LAYER:
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
def init_conv_layer(original_conv, weight_device, filter_indices_removal_ordered, layer_role, src_layer):
    num_filters_to_remove = len(filter_indices_removal_ordered)
    assert layer_role == DST_LAYER or layer_role == SRC_LAYER
    original_in_channel = original_conv.in_channels
    original_out_channel = original_conv.out_channels
    if layer_role == SRC_LAYER:
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



def populate_pruned_conv_src_consumer_layers(src_conv, new_src_conv, old_consumers, new_consumers,
                                             filter_removal_indices_ordered, weight_device):
    populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered, weight_device, SRC_LAYER,
                                    LAYER_WEIGHT)
    if src_conv.bias is not None:
        populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered,
                                        weight_device, SRC_LAYER, LAYER_BIAS)

    for old_consumer_conv, new_consumer_conv in zip(old_consumers, new_consumers):
        populate_pruned_conv_coe_tensor(old_consumer_conv, new_consumer_conv, filter_removal_indices_ordered,
                                        weight_device, DST_LAYER, LAYER_WEIGHT)
        if old_consumer_conv.bias is not None:
            populate_pruned_conv_coe_tensor(old_consumer_conv, new_consumer_conv, filter_removal_indices_ordered,
                                            weight_device, DST_LAYER, LAYER_BIAS)
# end of conv stuff



def get_coe_handle(conv, coe_type):
    if coe_type == LAYER_WEIGHT:
        return conv.weight.data.cpu().numpy()
    elif coe_type == LAYER_BIAS:
        return conv.bias.data.cpu().numpy()
    else:
        assert False, 'unrecognized coefficient type in conv layers'


def commit_coe(conv, coefficients, target_device, coe_type):
    if target_device.type == GPU_TARGET:
        if coe_type == LAYER_WEIGHT:
            conv.weight.data = torch.from_numpy(coefficients).cuda()
        elif coe_type == LAYER_BIAS:
            conv.bias.data = torch.from_numpy(coefficients).cuda()
        else:
            assert False, 'unrecognized coefficient type in conv layers'
