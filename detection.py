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


def main(config):
    #image_for_models
    img = cv2.imread(config.input_path)
    ie = IECore()

    #scratch_detection
    
    net_1 = ie.read_network(model=config.scratch_model)
    exec_net_1 = ie.load_network(network=net_1, device_name="CPU")
    
    #prepare_info
    data_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_data = data_gray.astype(np.float32) - 127.5
    norm_data /= 127.5
    resized_data = cv2.resize(norm_data, (608, 512))
    scratch_input = np.array([[resized_data]])
    out = exec_net_1.infer(inputs = {'input': scratch_input})
    
    mask = np.squeeze(out['output'])
    mask[mask >= 0.5] = 1.
    mask[mask < 0.5] = 0.
    #res_mask = mask.copy()
    h,w, = mask.shape
    for i in range(int(h/2-200),int(h/2-140),1):
        for j in range(int(w/2-30),int(w/2+30),1):
            mask[i,j]=1
           #if mask[i,j] == 1:
           #     for y in range(-5,5,1):
           #         for x in range(-5,5,1):
           #             res_mask[i+y,j+x]=1
    #mask = res_mask
    cv2.imwrite(config.output_dir+"\\mask\\8.png", mask*255)
    
    #inpainting
    if mask.max() == 1:
        net_2 = ie.read_network(model=config.inpainting_model)
        exec_net_2 = ie.load_network(network=net_2, device_name="CPU")
    
        #prepare_info
        resized_img = cv2.resize(img, (680, 512)).astype(np.float32)
        resized_mask = cv2.resize(mask, (680, 512)).astype(np.float32)

        resized_mask[resized_mask >= 0.5] = 1
        resized_mask[resized_mask < 0.5] = 0
        if len(resized_mask.shape) == 2:
            resized_mask = np.expand_dims(resized_mask, -1)
        resized_img = resized_img * (1 - resized_mask) + 255 * resized_mask
        inpainting_input = np.expand_dims(resized_img.transpose(2, 0, 1), 0)
        inpainting_mask = np.expand_dims(resized_mask.transpose(2, 0, 1), 0)
        print(inpainting_mask.shape)

        out = exec_net_2.infer(inputs = {'Placeholder': inpainting_input, 'Placeholder_1': inpainting_mask})
        print(next(iter(net_2.outputs)))
        out_img = out['Minimum'][0].transpose((1, 2, 0)).astype(np.uint8)
    
        #save_image
        cv2.imwrite(config.output_dir+"\\result\\8.png", out_img)
        cv2.imshow("input", cv2.resize(img, (680, 512), interpolation = cv2.INTER_CUBIC))
        cv2.imshow("out_img", out_img)
    cv2.imshow("mask", mask*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--GPU", type=int, default=-1)
        parser.add_argument("--scratch_model", type=str, default="scratch_detector.xml")
        parser.add_argument("--input_path", type=str, default="my\\8.jpeg")
        parser.add_argument("--output_dir", type=str, default="restored_image")
        parser.add_argument("--input_size", type=str, default="scale_256", help="resize_256|full_size|scale_256")
        parser.add_argument("--inpainting_model", type=str, default="gmcnn-places2-tf.xml")
        config = parser.parse_args()
        main(config)
