import numpy as np

# we might need to remove multiple elements in the specified dimension for every filter_index
def generate_trimed_weight_at_axis(old_weight_raw, filter_indices_for_removal, axis=0, slot_multiplier=1):
    # we transpose and transpose back
    weight_dims = old_weight_raw.ndim
    original_dim_list = list(range(weight_dims))
    tp_dim_list = original_dim_list
    tp_dim_list[axis] = 0
    tp_dim_list[0] = axis

    filter_indices_for_removal_mult = [i*slot_multiplier for i in filter_indices_for_removal]

    old_weight = np.transpose(old_weight_raw, tp_dim_list)
    weight_splitted = np.split(old_weight, filter_indices_for_removal_mult)
    # remove the first slice from every piece except the first one
    weight_tail_trimmed = [arr[slot_multiplier:] for arr in weight_splitted[1:]]
    new_weight_raw = np.concatenate((weight_splitted[0], np.concatenate(weight_tail_trimmed)))

    new_weight = np.transpose(new_weight_raw, tp_dim_list)
    return new_weight


