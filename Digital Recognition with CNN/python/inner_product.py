import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape
    n = param["w"].shape[1]

    ###### Fill in the code here ######

    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": np.zeros((n, k)) # replace 'data' value with your implementation
    }
    for i in range(k):
        output["data"][:, i] = np.dot(param["w"].T, input["data"][:, i]) + param["b"]
    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.

    # input_od = np.zeros_like(input_data["data"])

    # d, k = input_data["data"].shape
    # n = param["w"].shape[1]

    # for i in range(k):
    #     temp = input_data["data"][:, i]
    #     param_grad["w"] += np.outer(output["diff"][:, i], temp)
    #     param_grad["b"] += output["diff"][:, i]
    #     temp1 = np.outer(output["diff"][:, i], param["w"])
    #     temp2 = np.sum(temp1, axis=0)
    #     input_od[:, i] = temp2

    param_grad = {
        "w": np.zeros_like(param["w"]),
        "b": np.zeros_like(param["b"])
    }

    # Calculate input_od using matrix multiplication
    
    input_od = np.dot(param["w"], output["diff"])
    #the result is i-input feature to the j-input batch

    d, k = input_data["data"].shape 

    n = param["w"].shape[1]

    for i in range(k):
        temp = input_data["data"][:, i]
        param_grad["w"] += np.outer(temp, output["diff"][:, i])
        param_grad["b"] += output["diff"][:, i]

    return param_grad, input_od