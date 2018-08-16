from collections import OrderedDict
import torch.nn as nn
import numpy as np

from pytorch_trim_utils import SRC_LAYER, DST_LAYER
from pytorch_trim_utils.common_utils import generate_trimed_weight_at_axis
from pytorch_trim_utils.layer_pruning_utils import populate_pruned_conv_src_consumer_layers, init_pruned_layer


def duplicate_prune_conv_src_consumer_layers(src_conv, consumers, weight_device, filter_indices_removal_ordered):
    """
    Create a pruned version the src ocnv layer and the consumer conv layers,
    :param src_conv:
    :param consumers:
    :param weight_device:
    :param filter_indices_removal_ordered:
    :return: the new producer conv layer and the list of new consumers of this new conv layer
    """
    src_conv_dup = init_pruned_layer(src_conv, weight_device, filter_indices_removal_ordered, SRC_LAYER)

    consumers_dup = []
    for consumer in consumers:
        cur_consumer_dup = init_pruned_layer(consumer, weight_device, filter_indices_removal_ordered,
                                             DST_LAYER, src_conv)
        cur_consumer_dup.to(weight_device)
        consumers_dup.append(cur_consumer_dup)

    populate_pruned_conv_src_consumer_layers(src_conv, src_conv_dup, consumers, consumers_dup,
                                             filter_indices_removal_ordered, weight_device)
    return src_conv_dup, consumers_dup


def pruning_conv_filters(src_conv, consumer_convs, weight_device, selector, **kwarg):
    """
    Prune a conv layer at filter granularity, and also update its downstream consumers
    (Right now it only supports conv/linear layers as consumers)
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
