
def get_conv2d_kernel_bias(layer_name, var_collections):
    """
    From a collection of variables, grab the variable corresponding to the specified conv layer
    :param layer_name: name of the target conv layer
    :param var_collections: collection of variables
    :return: kernel and bias variables
    """
    kernel_name = layer_name + '/kernel:0'
    bias_name = layer_name + '/bias:0'
    kernel_var = filter(lambda x: x.name == kernel_name, var_collections)
    bias_var = filter(lambda x: x.name == bias_name, var_collections)
    return kernel_var, bias_var


def generate_trimed_conv_graph(layers_to_trim, old_graph, select_fun):
    """
    From an old graph, create a new graph, the subset of layers in the given list would be trimmed according to the
    select_fun, who takes in the old graph and a layer name, produces the filter index to trim.
    :param layers_to_trim:
    :param old_graph:
    :param select_fun:
    :return: the new graph
    """
    # we first find the affected layer and the kernel and bias value
    # we would need to replace it with the value multiply by a mask
    for op in old_graph.get_operations():
        print(op)


def traverse_existing_graph():
    pass


def load_ckpt():
    """
    loads the checkpoint for subsequent retraining or graph manipulation
    :return:
    """
    pass

