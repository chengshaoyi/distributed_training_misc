import torch
from torch import nn as nn

from pytorch_trim_utils.common_utils import generate_trimed_weight_at_axis
from pytorch_trim_utils import SRC_LAYER, DST_LAYER, GPU_TARGET, CPU_TARGET

CONV_WEIGHT = 0
CONV_BIAS = 1

def populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered,
                                    weight_device, layer_role,
                                    coe_type):
    assert layer_role == DST_LAYER or layer_role == SRC_LAYER, 'unrecognized trim dim'
    assert coe_type == CONV_WEIGHT or coe_type == CONV_BIAS, 'unrecognized coe type'
    original_conv_coe = get_conv_coe_handle(src_conv, coe_type)
    new_conv_coe = get_conv_coe_handle(new_src_conv, coe_type)
    if layer_role == SRC_LAYER:
        original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_conv_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=0)
    elif layer_role == DST_LAYER and coe_type == CONV_WEIGHT:
        original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_conv_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=1)
    else:
        original_conv_coe_trimmed = original_conv_coe
    new_conv_coe[...] = original_conv_coe_trimmed
    commit_conv_coe(new_src_conv, new_conv_coe, weight_device, coe_type)




def init_pruned_layer(layer, weight_device, filter_indices_removal_ordered, layer_role):
    assert layer_role == DST_LAYER or layer_role == SRC_LAYER
    if isinstance(layer, nn.Conv2d):
        layer_init_fn = init_conv_layer
    else:
        assert False, "unimplemented"
    return layer_init_fn(layer,weight_device,filter_indices_removal_ordered,layer_role)



# CONV layer stuff
def init_conv_layer(src_conv, weight_device, filter_indices_removal_ordered, layer_role):
    num_filters_to_remove = len(filter_indices_removal_ordered)
    assert layer_role == DST_LAYER or layer_role == SRC_LAYER
    src_in_channel = src_conv.in_channels
    src_out_channel = src_conv.out_channels
    if layer_role == SRC_LAYER:
        src_out_channel -= num_filters_to_remove
    else:
        src_in_channel -= num_filters_to_remove
    src_conv_dup = nn.Conv2d(in_channels=src_in_channel, out_channels=src_out_channel,
                             kernel_size=src_conv.kernel_size, stride=src_conv.stride, padding=src_conv.padding,
                             dilation=src_conv.dilation, groups=src_conv.groups, bias=src_conv.bias is not None) \
        .to(weight_device)
    return src_conv_dup


def get_conv_coe_handle(conv, coe_type):
    if coe_type == CONV_WEIGHT:
        return conv.weight.data.cpu().numpy()
    elif coe_type == CONV_BIAS:
        return conv.bias.data.cpu().numpy()
    else:
        assert False, 'unrecognized coefficient type in conv layers'


def commit_conv_coe(conv, coefficients, target_device, coe_type):
    if target_device.type == GPU_TARGET:
        if coe_type == CONV_WEIGHT:
            conv.weight.data = torch.from_numpy(coefficients).cuda()
        elif coe_type == CONV_BIAS:
            conv.bias.data = torch.from_numpy(coefficients).cuda()
        else:
            assert False, 'unrecognized coefficient type in conv layers'


def populate_pruned_conv_src_consumer_layers(src_conv, new_src_conv, old_consumers, new_consumers,
                                             filter_removal_indices_ordered, weight_device):
    populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered, weight_device, SRC_LAYER,
                                    CONV_WEIGHT)
    if src_conv.bias is not None:
        populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered,
                                        weight_device, SRC_LAYER, CONV_BIAS)

    for old_consumer_conv, new_consumer_conv in zip(old_consumers, new_consumers):
        populate_pruned_conv_coe_tensor(old_consumer_conv, new_consumer_conv, filter_removal_indices_ordered,
                                        weight_device, DST_LAYER, CONV_WEIGHT)
        if old_consumer_conv.bias is not None:
            populate_pruned_conv_coe_tensor(old_consumer_conv, new_consumer_conv, filter_removal_indices_ordered,
                                            weight_device, DST_LAYER, CONV_BIAS)
