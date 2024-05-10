import numpy as np

def relu_forward(input_data):
    #print("Keys in input_data:", input_data.keys())

    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }
    #print(output)
    #Replace the following line with your implementation.
    output['data'] = np.maximum(input_data['data'], 0)

    return output

def relu_backward(output, input_data, layer):
    relu = np.maximum(input_data['data'], 0)
    applied = relu == input_data['data']
    input_od = output['diff'] * applied
    
    return input_od
