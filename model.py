import mimetypes
import os
from uuid import uuid4

import numpy as np
import triton_python_backend_utils as pb_utils

from inference import infer, initialize


class TritonPythonModel:
    def initialize(self, args):
        os.chdir(os.path.dirname(__file__))
        initialize()

    def execute(self, requests):
        responses = []

        for request in requests:
            source_inp_tensor = pb_utils.get_input_tensor_by_name(request, "SOURCE_INPUT")
            source_inp_bytes = b''.join(source_inp_tensor.as_numpy())

            source_mime_tensor = pb_utils.get_input_tensor_by_name(request, "SOURCE_MIME")
            source_mime_str = b''.join(source_mime_tensor.as_numpy()).decode('utf-8')

            target_inp_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_INPUT")
            target_inp_bytes = b''.join(target_inp_tensor.as_numpy())

            target_mime_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_MIME")
            target_mime_str = b''.join(target_mime_tensor.as_numpy()).decode('utf-8')

            source_extension = mimetypes.guess_extension(source_mime_str)
            target_extension = mimetypes.guess_extension(target_mime_str)
            
            if source_extension == None or target_extension == None:
                raise ValueError('Cannot map input mime types to extensions')
            
            source_filename = str(uuid4()) + source_extension
            target_filename = str(uuid4()) + target_extension
            
            with open(source_filename, 'wb') as source_file:
                source_file.write(source_inp_bytes)

            with open(target_filename, 'wb') as target_file:
                target_file.write(target_inp_bytes)

            output_path = infer(source_filename, target_filename)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            with open(output_path, 'rb') as output_file:
                output_bytes = output_file.read()

            output_arr = np.array(output_bytes, dtype=np.bytes_)
            output_tensor = pb_utils.Tensor("OUTPUT", output_arr.reshape([1]))

            output_mime_tensor = pb_utils.Tensor("OUTPUT_MIME", target_mime_tensor.as_numpy())

            # Clean up after inference
            os.remove(source_filename)
            os.remove(target_filename)
            os.remove(output_path)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor, output_mime_tensor])

            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

