import numpy as np
from utils import im2col_conv_batch

def pooling_layer_forward(input, layer):
    """
    Forward pass for the pooling layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    """
    
    h_in = input['height']
    w_in = input['width']
    c = input['channel']
    batch_size = input['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    h_out = int((h_in + 2 * pad - k) / stride + 1)
    w_out = int((w_in + 2 * pad - k) / stride + 1)
    
    output = {}
    output['height'] = h_out
    output['width'] = w_out
    output['channel'] = c
    output['batch_size'] = batch_size
    output['data'] = np.zeros((h_out, w_out, c, batch_size)) # replace with your implementation

    ##### Fill in the code here ######
    features = np.reshape(input['data'], (h_in, w_in, c, batch_size), order='F')
    
    for i in range(batch_size):
        for j in range(c):
            f = features[:,:,j,i]
            for y in range(h_out):
                for x in range(w_out):
                    temp = f[y*stride:y*stride+k, x*stride:x*stride+k]
                    output['data'][y, x, j, i] = np.max(temp)

    output['data'] = np.reshape(output['data'], (h_out * w_out * c, batch_size), order='F')
    
    output = {
        'data': output['data'],
        'height': h_out,
        'width': w_out,
        'channel': c,
        'batch_size': batch_size
    }
    return output
# def pooling_layer_forward(input, layer):
#     h_in = input['height']
#     w_in = input['width']
#     c = input['channel']
#     batch_size = input['batch_size']
#     k = layer['k']
#     stride = layer['stride']
#     pad = layer['pad']

#     h_out = (h_in + 2*pad - k) // stride + 1
#     w_out = (w_in + 2*pad - k) // stride + 1

#     #col_input = im2col_conv_batch(input, layer, h_out, w_out)
#     #print("Shape of col_input:", col_input.shape) # Debugging print
#     #col_input_reshaped = col_input.reshape(k * k, c, h_out * w_out, batch_size)
#     #max_values = np.max(col_input_reshaped, axis=0)
#     #print(max_values.shape)

#     output_data = max_values.reshape(h_out* w_out* c, batch_size, order='F')

#     #print(output_data.shape)
    
#     output = {
#         'data': output_data,
#         'height': h_out,
#         'width': w_out,
#         'channel': c,
#         'batch_size': batch_size
#     }

#     return output

# def pooling_layer_forward(input, layer):
#     h_in = input['height']
#     w_in = input['width']
#     c = input['channel']
#     batch_size = input['batch_size']
#     k = layer['k']
#     stride = layer['stride']
#     pad = layer['pad']

#     h_out = (h_in + 2*pad - k) // stride + 1
#     w_out = (w_in + 2*pad - k) // stride + 1
   
#     output_data = np.zeros((h_out * w_out * c, batch_size)) 

#     # Convert entire batch to column format
#     col_input_all = im2col_conv_batch(input, layer, h_out, w_out)
    
#     for i in range(batch_size):
#         # Extract the columns corresponding to the current image in the batch
#         col_input = col_input_all[:, :, i]
        
#         # Reshape the column data for max pooling
#         col_input_reshaped = col_input.reshape(k * k, c, h_out * w_out)
        
#         # Apply max pooling on the reshaped column data
#         max_values = np.max(col_input_reshaped, axis=0)

#         # Store max values in the output data
#         output_data[:, i] = max_values.ravel()

#     output = {
#         'data': output_data,
#         'height': h_out,
#         'width': w_out,
#         'channel': c,
#         'batch_size': batch_size
#     }

#     return output



def pooling_layer_backward(output, input, layer):
    """
    Backward pass for the pooling layer.

    Parameters:
    - output (dict): Contains the gradients from the next layer.
    - input (dict): Contains the original input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.

    Returns:
    - input_od (numpy.ndarray): Gradient with respect to the input.
    """

    h_in = input['height']
    w_in = input['width']
    c = input['channel']
    batch_size = input['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    h_out = (h_in + 2*pad - k) // stride + 1
    w_out = (w_in + 2*pad - k) // stride + 1

    input_od = np.zeros(input['data'].shape)
    input_od = input_od.reshape(h_in * w_in * c * batch_size, 1)

    im_b = np.reshape(input['data'], (h_in, w_in, c, batch_size), order='F')
    im_b = np.pad(im_b, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='constant')
    
    diff = np.reshape(output['diff'], (h_out*w_out, c*batch_size), order='F')

    for h in range(h_out):
        for w in range(w_out):
            matrix_hw = im_b[h*stride : h*stride + k, w*stride : w*stride + k, :, :]
            flat_matrix = matrix_hw.reshape((k*k, c*batch_size), order='F')
            i1 = np.argmax(flat_matrix, axis=0)
            R, C = np.unravel_index(i1, matrix_hw.shape[:2], order='F')
            nR = h*stride + R
            nC = w*stride + C
            i2 = np.ravel_multi_index((nR, nC), (h_in, w_in), order='F')
            i4 = np.ravel_multi_index((i2, np.arange(c*batch_size)), (h_in*w_in, c*batch_size), order='F')
            i3 = np.ravel_multi_index((h, w), (h_out, w_out), order='F')
            input_od[i4] += diff[i3:i3+1, :].T

    input_od = np.reshape(input_od, (h_in*w_in, c*batch_size), order='F')
    input_od = np.reshape(input_od, (h_in*w_in*c, batch_size), order='F')

    return input_od
