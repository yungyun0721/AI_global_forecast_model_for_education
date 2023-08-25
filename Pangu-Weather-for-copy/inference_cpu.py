import os
import numpy as np
import onnx
import onnxruntime as ort


# The directory of your input and output data
input_data_dir = 'input_data'# add inital time
output_data_dir = 'output_data' # add you name
model_24 = onnx.load('pangu_weather_24.onnx')


if not os.path.isdir(output_data_dir):
    os.mkdir(output_data_dir)
# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=True
options.enable_mem_pattern = True
options.enable_mem_reuse = True
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 1 # how many cpu

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

# Initialize onnxruntime session for Pangu-Weather Models
ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=options, providers=['CPUExecutionProvider'])

# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

np.save(os.path.join(output_data_dir, 'output_upper_0'), input)
np.save(os.path.join(output_data_dir, 'output_surface_0'), input_surface)

for i in range(10):
    print(f'{(i+1)*24} hour')
    # Run the inference session
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})

    # Save the results
    np.save(os.path.join(output_data_dir, f'output_upper_{i+1}'), output)
    np.save(os.path.join(output_data_dir, f'output_surface_{i+1}'), output_surface)
    
    input = output
    input_surface = output_surface 
    
