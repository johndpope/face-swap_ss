# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
# import triton_python_backend_utils as pb_utils


import mimetypes
import os
from os.path import isfile, join, splitext, basename
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from insightface_func.face_detect_crop_mutil import Face_detect_crop
from models.models import create_model
from options.test_options import TestOptions
from util.videoswap import video_swap


model, app = None, None
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def initialize():
    opt = TestOptions()
    opt.initialize()
    opt.parser.add_argument('-f')  # dummy arg to avoid bug
    opt = opt.parse()
    opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
    opt.isTrain = False
    torch.nn.Module.dump_patches = True
    global model
    model = create_model(opt)
    model.eval()
    global app
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(256, 256))


def infer(source, target, output_path, result_dir = './output', crop_size=224):
    assert isfile(source), f'Can\'t find source at {source}'
    assert isfile(target), f'Can\'t find target at {target}'
    output_filename = f'infer-{splitext(basename(source))[0]}-{splitext(basename(target))[0]}.mp4'
    output_path = join(result_dir, output_filename)

    assert model is not None
    assert app is not None

    img_a_whole = cv2.imread(source)
    img_a_align_crop, _ = app.get(img_a_whole, crop_size)
    img_a_align_crop_pil = Image.fromarray(
        cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
    img_a = transformer_Arcface(img_a_align_crop_pil)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
    img_id = img_id.cuda()

    img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
    latend_id = model.netArc(img_id_downsample)
    latend_id = latend_id.detach().to('cpu')
    latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
    latend_id = latend_id.to('cuda')

    video_swap(target, latend_id, model, app, output_path)
    return output_path


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


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    initialize()
    infer('./demo_file/Example_Man_01.png',
          './demo_file/test_video_05.MOV',
          './output/demo.mp4')
