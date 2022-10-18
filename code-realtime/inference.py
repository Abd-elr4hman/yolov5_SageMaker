import torch
import os
import logging
import io
import json
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms

import tempfile
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def transform_fn(model, request_body, content_type, accept):
    
    device = get_device()
    
    # read image from io byte stream
    f = io.BytesIO(request_body)

    # create a temporary file and save the io byte image in it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    # Preprocess image
    batch_image = image_file_to_tensor(tfile.name)
    
    output = predict(batch_image.to('cpu') , model)
    prediction = output[0].detach().cpu().numpy().tolist()

    return json.dumps(prediction)

    
def predict(inputs, model):
    logger.info(">>> Invoking model!..")

    with torch.no_grad():
        #device = get_device()
        device= 'cpu'
        model = model.to(device)
        input_data = inputs.to(device)
        model.eval()
        outputs = model(input_data)

    return outputs


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def image_file_to_tensor(path):
    # When the image file is read with the OpenCV function imread(), the order of colors is BGR
    cv_img = cv2.imread(path,1).astype('uint8')

    cv_img = cv2.resize(cv_img, (640, 640), interpolation = cv2.INTER_AREA)

    # Convert BGR to RGB
    cv_img = cv_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

    image = np.ascontiguousarray(cv_img)
    image = torch.from_numpy(image)

    # to float
    image = image.float()

    # normalise
    image /= 255

    # to batch image
    batch_image = image[None,:]
    return batch_image
