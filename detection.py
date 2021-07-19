import numpy as np
import cv2 
import sys
import os
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image, ImageFile

from openvino.inference_engine import IECore
import argparse

def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    print(w, h)
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")

def main(config):
    #models_shape
    scratch_h, scratch_w = (608,512)
    inp_h, inp_w = (680, 512)
    
    #image_for_models
    input = cv2.imread(config.input_path)
    
    ie = IECore()

    #scratch_detection
    model = config.model
    net = ie.read_network(model=model)
    assert len(net.input_info.keys()) == 1
    assert len(net.outputs) == 1
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.reshape({input_blob: (1, 1, scratch_w, scratch_h)})          
    if config.GPU >= 0:
        exec_net = ie.load_network(network=net, device_name="GPU")
    else:
        exec_net = ie.load_network(network=net, device_name="CPU")

    #prepare_info
    scratch_image = os.path.join(config.input_path)
    scratch_image = Image.open(scratch_image).convert("RGB")
    w, h = scratch_image.size
    transformed_image_PIL = scratch_image.resize((scratch_h, scratch_w), Image.BICUBIC)
    scratch_image = transformed_image_PIL.convert("L")
    scratch_image = tv.transforms.ToTensor()(scratch_image)
    scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
    scratch_image = torch.unsqueeze(scratch_image, 0)
    print(scratch_image.shape)
    
    with torch.no_grad():
        scratch_res = exec_net.infer(inputs={input_blob: scratch_image})
        scratch_res = scratch_res[out_blob]
        
        mask = scratch_res[0]
        mask = mask.transpose((1, 2, 0))
        print(1)
        scratch_output = cv2.resize(input, (scratch_h, scratch_w), interpolation = cv2.INTER_CUBIC)
        for i in range(scratch_w):
            for j in range (scratch_h):
                scratch_res[0,0,i,j] = 1/(1+np.exp(-mask[i,j,0]))
                if scratch_res[0,0,i,j]>0.4:
                    scratch_res[0,0,i,j] = 1
                    scratch_output[i,j] = [255,255,255]
                else:
                    scratch_res[0,0,i,j] = 0
                    scratch_output[i,j] = [0,0,0]
        print(2)
        #save_mask
        cv2.imwrite(config.output_dir+"\\mask\\1.png", scratch_output)
    print(3)
    #inpainting
    inp_model = config.inpainting_model
    inp_net = ie.read_network(model=inp_model)
    inp_input_blob = next(iter(inp_net.input_info))
    inp_out_blob = next(iter(inp_net.outputs))
    if config.GPU >= 0:
        inp_exec_net = ie.load_network(network=inp_net, device_name="GPU")
    else:
        inp_exec_net = ie.load_network(network=inp_net, device_name="CPU")
    print(4)
    #prepare_info
    #inpainting_input
    inp_input = cv2.resize(input, (inp_h, inp_w), interpolation = cv2.INTER_CUBIC)
    inp_input = inp_input.transpose((2, 0, 1))
    #mask
    inp_mask = scratch_res
    inp_mask = inp_mask[0,0]
    inp_mask = cv2.resize(inp_mask, (inp_h, inp_w), interpolation=cv2.INTER_CUBIC)
    inp_mask=inp_mask.astype(int)
    inp_mask = [[inp_mask]]
    print(5)
    inp_res = inp_exec_net.infer(inputs={inp_input_blob: inp_input, inp_input_blob+"_1": inp_mask})
    inp_res = inp_res[inp_out_blob]
    inp_res = inp_res[0].transpose((1, 2, 0))
    print(6)
    #save_image
    cv2.imwrite(config.output_dir+"\\result\\1.png", inp_res)
    print(7)
    cv2.imshow("input", cv2.resize(input, (inp_h, inp_w), interpolation = cv2.INTER_CUBIC))
    cv2.imshow("mask", scratch_output)
    cv2.imshow("result", inp_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--GPU", type=int, default=-1)
        parser.add_argument("--model", type=str, default="scratch_detector.xml")
        parser.add_argument("--input_path", type=str, default="my\\2.png")
        parser.add_argument("--output_dir", type=str, default="restored_image")
        parser.add_argument("--input_size", type=str, default="scale_256", help="resize_256|full_size|scale_256")
        parser.add_argument("--inpainting_model", type=str, default="gmcnn-places2-tf.xml")
        config = parser.parse_args()
        main(config)
