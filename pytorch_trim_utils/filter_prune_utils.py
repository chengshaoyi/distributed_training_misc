from collections import OrderedDict
import torch.nn as nn
import numpy as np
import torch

CPU_TARGET = "cpu"
GPU_TARGET = "cuda"

# enum for conv coefficients
CONV_WEIGHT = 0
CONV_BIAS = 1

TRIM_OUTER = 0
TRIM_INNER = 1


def populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered,
                                    weight_device, trim_dim,
                                    coe_type):
    assert trim_dim == TRIM_INNER or trim_dim == TRIM_OUTER, 'unrecognized trim dim'
    assert coe_type == CONV_WEIGHT or coe_type == CONV_BIAS, 'unrecognized coe type'
    original_conv_coe = get_conv_coe_handle(src_conv, coe_type)
    new_conv_coe = get_conv_coe_handle(new_src_conv, coe_type)
    if trim_dim == TRIM_OUTER:
        original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_conv_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=0)
    elif trim_dim == TRIM_INNER and coe_type == CONV_WEIGHT:
        original_conv_coe_trimmed = generate_trimed_weight_at_axis(original_conv_coe,
                                                                   filter_removal_indices_ordered,
                                                                   axis=1)
    else:
        original_conv_coe_trimmed = original_conv_coe
    new_conv_coe[...] = original_conv_coe_trimmed
    commit_conv_coe(new_src_conv, new_conv_coe, weight_device, coe_type)


def populate_pruned_conv_src_consumer_layers(src_conv, new_src_conv, old_consumers, new_consumers,
                                             filter_removal_indices_ordered, weight_device):
    populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered, weight_device, TRIM_OUTER,
                                    CONV_WEIGHT)
    if src_conv.bias is not None:
        populate_pruned_conv_coe_tensor(src_conv, new_src_conv, filter_removal_indices_ordered,
                                        weight_device, TRIM_OUTER, CONV_BIAS)

    for old_consumer_conv, new_consumer_conv in zip(old_consumers, new_consumers):
        populate_pruned_conv_coe_tensor(old_consumer_conv, new_consumer_conv, filter_removal_indices_ordered,
                                        weight_device, TRIM_INNER, CONV_WEIGHT)
        if old_consumer_conv.bias is not None:
            populate_pruned_conv_coe_tensor(old_consumer_conv, new_consumer_conv, filter_removal_indices_ordered,
                                            weight_device, TRIM_INNER, CONV_BIAS)




def duplicate_prune_conv_src_consumer_layers(src_conv, consumers, weight_device, filter_indices_removal_ordered):
    """
    Create a pruned version the src ocnv layer and the consumer conv layers,
    :param src_conv:
    :param consumers:
    :param weight_device:
    :param filter_indices_removal_ordered:
    :return: the new producer conv layer and the list of new consumers of this new conv layer
    """
    # for now we just delete filters from conv2d layers
    # we migh want to accommodate dw conv
    assert isinstance(src_conv, nn.Conv2d)
    assert all(isinstance(consumer, nn.Conv2d) for consumer in consumers)
    num_filters_to_remove = len(filter_indices_removal_ordered)
    src_in_channel = src_conv.in_channels
    src_out_channel = src_conv.out_channels

    src_conv_dup = nn.Conv2d(in_channels=src_in_channel, out_channels=src_out_channel - num_filters_to_remove,
                             kernel_size=src_conv.kernel_size, stride=src_conv.stride, padding=src_conv.padding,
                             dilation=src_conv.dilation, groups=src_conv.groups, bias=src_conv.bias is not None)\
        .to(weight_device)

    consumers_dup = [
        nn.Conv2d(in_channels=consumer.in_channels - num_filters_to_remove, out_channels=consumer.out_channels,
                  kernel_size=consumer.kernel_size, stride=consumer.stride, padding=consumer.padding,
                  dilation=consumer.dilation, groups=consumer.groups, bias=consumer.bias is not None)
        for consumer in consumers]
    for consumer_dup in consumers_dup:
        consumer_dup.to(weight_device)
    populate_pruned_conv_src_consumer_layers(src_conv,src_conv_dup,consumers,consumers_dup,
                                             filter_indices_removal_ordered,weight_device)

    return src_conv_dup, consumers_dup


def generate_trimed_weight_at_axis(old_weight_raw, filter_indices_for_removal, axis=0):
    # we transpose and transpose back
    weight_dims = old_weight_raw.ndim
    original_dim_list = list(range(weight_dims))
    tp_dim_list = original_dim_list
    tp_dim_list[axis] = 0
    tp_dim_list[0] = axis
    old_weight = np.transpose(old_weight_raw, tp_dim_list)
    weight_splitted = np.split(old_weight, filter_indices_for_removal)
    # remove the first slice from every piece except the first one
    weight_tail_trimmed = [arr[1:] for arr in weight_splitted[1:]]
    new_weight_raw = np.concatenate((weight_splitted[0], np.concatenate(weight_tail_trimmed)))
    new_weight = np.transpose(new_weight_raw, tp_dim_list)
    return new_weight


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


def pruning_conv_filters(src_conv, consumer_convs, weight_device, selector, **kwarg):
    """
    Prune a conv layer at filter granularity, and also update its downstream consumers
    (Right now it only supports conv layers as consumers)
    which filters to erase depends on the selector function.
    :param src_conv: producer conv
    :param consumer_convs: [list of consumer layers for the producer conv]
    :param weight_device: where does the layer/weight live (cpu or cuda)
    :param selector: return a list of integers, indicating which filter to throw away in the conv layer
    :param kwarg: arguments to selector
    :return: new tuple of (producer conv, [list of consumer layers for the conv])
    """
    filter_indices_to_remove = selector(src_conv, **kwarg)
    filter_removal_indices_cleaned = list(OrderedDict.fromkeys(filter_indices_to_remove))
    filter_removal_indices_ordered = sorted(filter_removal_indices_cleaned)

    new_src_conv, new_consumers = duplicate_prune_conv_src_consumer_layers(src_conv, consumer_convs, weight_device,
                                                                           filter_removal_indices_ordered)
    return new_src_conv, new_consumers


def main():
    print("test")
    '''old_weight_0 = np.arange(0,10).reshape([1,10])
    old_weight_1 = np.arange(0,10).reshape([1,10])
    old_weight = np.concatenate((old_weight_0,old_weight_1))
    old_weight = np.transpose(old_weight,[1,0])
    print(old_weight)
    print(old_weight.shape)
    print("-----------")
    new_weight = generate_trimed_weight_outer(old_weight, filter_indices_for_removal)
    print(new_weight)
    print(new_weight.shape)
    print("============")'''
    old_weight_0 = np.arange(0, 10).reshape([1, 10])
    old_weight_1 = np.arange(10, 20).reshape([1, 10])
    old_weight_2 = np.arange(20, 30).reshape([1, 10])
    old_weight_3 = np.arange(30, 40).reshape([1, 10])
    old_weight = np.concatenate((old_weight_0, old_weight_1, old_weight_2, old_weight_3))
    print(old_weight)
    filter_indices_for_removal = [2, 3]
    new_weight = generate_trimed_weight_at_axis(old_weight, filter_indices_for_removal)
    print(new_weight)
    # old_weight_tr = np.transpose(old_weight,[1,0])
    print(generate_trimed_weight_at_axis(old_weight, filter_indices_for_removal, 1))


if __name__ == '__main__':
    main()
