import os
import cv2
import sys
from glob import glob
import argparse
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore




def process_single_image_super_resolution_1032(source, net, exec_net, ie=None, model=None, d=None):
    k = 4
    if not exec_net:
        net = ie.read_network(model=model)
        net.reshape({"0":(1, 3, source.shape[0], source.shape[1]), "1":(1, 3, source.shape[0]*k, source.shape[1]*k)})
        exec_net = ie.load_network(network=net, device_name=d)
    out_blob = next(iter(net.outputs))
    in1 = [source.transpose((2, 0, 1))]
    in2_cubic = cv2.resize(source, (source.shape[1] * k, source.shape[0] * k),interpolation=cv2.INTER_CUBIC)
    in2 = [in2_cubic.transpose((2, 0, 1))]
    # res
    output = exec_net.infer(inputs = {'0': in1, "1": in2})
    output = output[out_blob][0]
    output = output.transpose(1,2,0)
    return output,in2_cubic, net, exec_net


def process_single_image_super_resolution_1033(source, net, exec_net, ie=None, model=None, d=None):
    k = 3
    if not exec_net:
        net = ie.read_network(model=model)
        net.reshape({"0":(1, 3, source.shape[0], source.shape[1]), "1":(1, 3, source.shape[0]*k, source.shape[1]*k)})
        exec_net = ie.load_network(network=net, device_name=d)
    out_blob = next(iter(net.outputs))
    in1 = [source.transpose((2, 0, 1))]
    in2_cubic = cv2.resize(source, (source.shape[1] * k, source.shape[0] * k),interpolation=cv2.INTER_CUBIC)
    in2 = [in2_cubic.transpose((2, 0, 1))]
    # res
    output = exec_net.infer(inputs = {'0': in1, "1": in2})
    output = output[out_blob][0]
    output = output.transpose(1,2,0)
    return output,in2_cubic, net, exec_net

def process_pytorch_example():
    pass

def process_RCAN():
    pass

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        images folder', required=True, type=str)
    # parser.add_argument('-l', '--cpu_extension', help='MKLDNN \
    #     (CPU)-targeted custom layers.Absolute path to a shared library \
    #     with the kernels implementation', type=str, default=None)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('--height', help='Height of the image', default=0, type=int)
    parser.add_argument('--width', help='Width of the image', default=0, type=int)
    return parser


def resizeImages():
    inputs = glob("images" + "/*")
    for imgname in inputs:
        img = cv2.imread(imgname)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_NEAREST)
        # cv2.imshow(imgname, img)
        cv2.imwrite("converted/"+imgname[imgname.rfind("\\")+1:],img)

# resizeImages()

# можно отсортировать изображения так, чтобы было минимальное количество изменений разрешений
def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    inputs = glob(args.input + "/*")
    ie = IECore()

    net = None
    exec_net = None
    if args.height != 0 and args.width != 0:
        net = ie.read_network(model=args.model)
        net.reshape({"0":(1, 3, args.width, args.height), "1":(1, 3, args.width*4, args.height*4)})
        exec_net = ie.load_network(network=net, device_name=args.device)

    iter = 0
    prevRes = [0,0]
    for i in inputs:
        iter +=1
        img = cv2.imread(i)
        res,cubic = None,None
        if prevRes[0] != np.shape(img)[0] or prevRes[1] != np.shape(img)[1]:
            print("new resolution")
            prevRes[0] = np.shape(img)[0]
            prevRes[1] = np.shape(img)[1]
            res,cubic, net,exec_net = process(img, None, None, ie, args.model, args.device)
        else:    
            res,cubic, net,exec_net = process(img, net, exec_net, ie, args.model, args.device)
        # resaults.add([res,cubic]])
        cv2.imshow("cubic interpolation " + str(iter), cubic)
        cv2.imshow("super resolution " + str(iter), res)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    sys.exit(main())