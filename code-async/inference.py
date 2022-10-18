import time
import os
import io
import json
import tempfile

import logging

import math
import numpy as np

import torch
import torchvision
import cv2


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def transform_fn(model, request_body, content_type, accept):
    
    interval = int(1)
    batch_size = int(10)


    device = get_device()
    
    # read image from io byte stream
    f = io.BytesIO(request_body)

    # create a temporary file and save the io byte image in it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    all_predictions = []

    for batch_frames in batch_generator(tfile, interval, batch_size):

        batch_inputs= preprocess(batch_frames)
        logger.info(">>> Length of batch inputs: %d" % len(batch_frames))
        print("len of batch_inputs {}".format(len(batch_frames)))

        batch_outputs = predict(batch_inputs, model)
        logger.info(">>> Length of batch predictions: %d" % len(batch_outputs))
        print("len of batch_outputs {}".format(len(batch_outputs)))

        pred_dict= postprocess(batch_outputs)

        all_predictions.append(pred_dict)

    return json.dumps(all_predictions)


    
def predict(inputs, model):
    logger.info(">>> Invoking model!..")

    with torch.no_grad():
        device = get_device()
        #device= 'cpu'
        model = model.to(device)
        input_data = inputs.to(device)
        model.eval()
        outputs = model(input_data)

    return outputs


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def preprocess(inputs):
    outputs = torch.stack(inputs)
    return outputs
    

def image_file_to_tensor(cv_img):
    # Resize
    cv_img = cv2.resize(cv_img, (640, 640), interpolation = cv2.INTER_AREA)

    # Convert BGR to RGB
    cv_img = cv_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

    image = np.ascontiguousarray(cv_img)
    image = torch.from_numpy(image)

    # to float
    image = image.float()

    # normalise
    image /= 255

    return image


def batch_generator(tfile, interval, batch_size):
    cap = cv2.VideoCapture(tfile.name)
    frame_index = 0
    frame_buffer = []

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            cap.release()
            if frame_buffer:
                yield frame_buffer
            return

        if frame_index % interval == 0:
            #frame_resized = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            frame_preprocessed= image_file_to_tensor(frame)
            frame_buffer.append(frame_preprocessed)

        if len(frame_buffer) == batch_size:
            yield frame_buffer
            frame_buffer.clear()

        frame_index += 1
    else:
        raise Exception("Failed to open video '%s'!.." % tfile.name)


def postprocess(batch_outputs):
    """
    Return postprocessed batch outputs.

    Arguments:
        batch_ouputs: tensor of shape (1,22500,85)

    Returns:
        batch list: list of outputs for each is for a single batch.
                    for each single batch a list of frame outputs.
                    for each single frame a list of detected_instance_dict.
                    [batch_number][frames_in_a_single_batch][instances_in_a_single_frame]
                    each single instance is as follows
                    {'box': [578.0349731445312,
                            286.77862548828125,
                            604.8016967773438,
                            372.55804443359375],
                        'conf': 0.250052809715271,
                        'cls': 0.0}

    """
    pred= non_max_suppression(batch_outputs)

    '''
    pred: list of tensors, each is of the form
    array([[ 4.19481430e+01,  2.31071228e+02,  1.93197388e+02, 5.39934509e+02,  8.99273098e-01,  0.00000000e+00],
       [ 5.30732117e+02,  2.29216156e+02,  6.39800842e+02, 5.19732544e+02,  8.79986107e-01,  0.00000000e+00],
       [ 1.75919220e+02,  2.42786438e+02,  2.72277313e+02, 5.08229858e+02,  8.02967429e-01,  0.00000000e+00],
       [-3.44982910e+00,  1.32060059e+02,  6.32543701e+02, 4.57163208e+02,  6.62323236e-01,  5.00000000e+00],
       [-1.29940033e-01,  3.25055237e+02,  5.67167854e+01, 5.17109070e+02,  5.19294977e-01,  0.00000000e+00]])

    where each row is a single object detected.
    '''
    batch_list=[]
    for tensor in pred:
        print(tensor.shape)
        frame_list=[]
        for *xyxy, conf, cls in reversed(tensor).tolist():

            instance_dict={
                'box': xyxy,
                'conf': conf,
                'cls': cls
            }
            frame_list.append(instance_dict)

        batch_list.append(frame_list)

    return batch_list



############################# Post processing utils ##############################

def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])
    
def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)
    
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        '''
        if (time.time() - t) > time_limit:
            #LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
        '''

    return output