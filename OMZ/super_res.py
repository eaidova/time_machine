import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore




def process(source, device, configPath):        
    k = 4
    ie = IECore()
    net = ie.read_network(model=configPath)
    out_blob = next(iter(net.outputs))
    in1 = [source.transpose((2, 0, 1))]
    in2_cubic = cv2.resize(source, (source.shape[1] * k, source.shape[0] * k),interpolation=cv2.INTER_CUBIC)
    in2 = [in2_cubic.transpose((2, 0, 1))]
    # print("preprocessed image1 shape",np.shape(in1))
    # net reshape
    net.reshape({"0":(1,3,source.shape[0],source.shape[1]), "1":(1,3,in2_cubic.shape[0],in2_cubic.shape[1])})
    exec_net = ie.load_network(network=net, device_name=device)
    # print("shape0", self.net.inputs['0'].shape)
    # res
    output = exec_net.infer(inputs = {'0': in1, "1": in2})
    output = output[out_blob][0]
    output = output.transpose(1,2,0)
    return output,in2_cubic


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    # parser.add_argument('-l', '--cpu_extension', help='MKLDNN \
    #     (CPU)-targeted custom layers.Absolute path to a shared library \
    #     with the kernels implementation', type=str, default=None)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE classification sample")

    # Read image
    img = cv2.imread(args.input)
        
    # Process image
    res,cubic = process(img, args.device, args.model)
    # print(res.shape)
    cv2.imshow("cubic interpolation", cubic)
    cv2.imshow("super resolution", res)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    sys.exit(main())