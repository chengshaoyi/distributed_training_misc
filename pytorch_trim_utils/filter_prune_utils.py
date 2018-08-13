from collections import OrderedDict
import torch.nn as nn
import numpy as np

def duplicate_conv_src_consumer_layers(conv_and_consumers, filter_indices_removal_ordered):
    src_conv = conv_and_consumers[0]
    consumers = conv_and_consumers[1]
    # for now we just delete filters
    assert isinstance(src_conv, nn.Conv2d)
    assert all(isinstance(consumer, nn.Conv2d) for consumer in consumers)
    num_filters_to_remove = len(filter_indices_removal_ordered)
    src_in_channel = src_conv.in_channels
    src_out_channel = src_conv.out_channels
    src_conv_dup = nn.Conv2d(in_channels=src_in_channel,out_channels=src_out_channel-num_filters_to_remove,
                             kernel_size=src_conv.kernel_size, stride=src_conv.stride, padding=src_conv.padding,
                             dilation= src_conv.dilation, groups=src_conv.groups,bias=src_conv.bias)

    consumers_dup = [nn.Conv2d(in_channels=consumer.in_channels-num_filters_to_remove, out_channels=consumer.out_channels,
                             kernel_size=consumer.kernel_size, stride=consumer.stride, padding=consumer.padding,
                             dilation= consumer.dilation, groups=consumer.groups,bias=consumer.bias)
                     for consumer in consumers]
    return src_conv_dup,consumers_dup



def generate_trimed_weight_outer(old_weight,filter_indices_for_removal):
    weight_splitted = np.split(old_weight, filter_indices_for_removal)
    # remove the first slice from every piece except the first one
    weight_tail_trimmed = [arr[1:] for arr in weight_splitted[1:]]
    new_weight = np.concatenate((weight_splitted[0],np.concatenate(weight_tail_trimmed)))
    return new_weight


def pruning_conv_filters(conv_and_consumers, weight_device, selector, **kwarg):
    """
    Prune a conv layer at filter granularity, and also update its downstream consumers
    (Right now it only supports conv layers as consumers)
    which filters to erase depends on the selector function.
    :param conv_and_consumers: a tuple of (producer conv, [list of consumer layers for the conv])
    :param selector: return a list of integers, indicating which filter to throw away in the conv layer
    :param kwarg: arguments to selector
    :return: new tuple of (producer conv, [list of consumer layers for the conv])
    """
    src_conv = conv_and_consumers[0]
    filter_indices_to_remove = selector(src_conv, kwarg)
    filter_removal_indices_cleaned = list(OrderedDict(x, True) for x in filter_indices_to_remove)
    filter_removal_indices_ordered = sorted(filter_removal_indices_cleaned)

    new_src_conv, new_consumers = duplicate_conv_src_consumer_layers(conv_and_consumers, filter_removal_indices_ordered)

    consumers = conv_and_consumers[1]
    original_conv_weight = src_conv.weight.data.cpu().numpy()


    #for filter_ind in filter_removal_indices_ordered:


def main():
    print "test"
    filter_indices_for_removal = [2,5,6]
    old_weight_0 = np.arange(0,10).reshape([1,10])
    old_weight_1 = np.arange(0,10).reshape([1,10])
    old_weight = np.concatenate((old_weight_0,old_weight_1))
    old_weight = np.transpose(old_weight,[1,0])
    print old_weight
    print old_weight.shape
    print "-----------"
    new_weight = generate_trimed_weight_outer(old_weight, filter_indices_for_removal)
    print new_weight
    print new_weight.shape


if __name__ == '__main__':
    main()