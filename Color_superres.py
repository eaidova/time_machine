import os
import cv2 as cv
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import argparse
import time
import sys
import logging as log
from glob import glob

def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    args.add_argument('-m1', '--model_1', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    args.add_argument('-m2', '--model_2', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    # parser.add_argument('-l', '--cpu_extension', help='MKLDNN \
    #     (CPU)-targeted custom layers.Absolute path to a shared library \
    #     with the kernels implementation', type=str, default=None)
    args.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    args.add_argument('-o', '--output', help='Path to output folder', default="output", type=str)

    return parser

def color_image(img, net_1, exec_net):

    

    input_blob = next(iter(net_1.input_info))
    input_shape = net_1.input_info[input_blob].input_data.shape
    assert input_shape[1] == 1, "Expected model input shape with 1 channel"

    inputs = {}
    for input_name in net_1.input_info:
        inputs[input_name] = np.zeros(net_1.input_info[input_name].input_data.shape)

    assert len(net_1.outputs) == 1, "Expected number of outputs is equal 1"
    output_blob = next(iter(net_1.outputs))
    output_shape = net_1.outputs[output_blob].shape

    _, _, h_in, w_in = input_shape

    imshow_size = (img.shape[1], img.shape[0])


    # Reading the frame
    (h_orig, w_orig) = img.shape[:2]
            
    # Prepare frame

    if img.shape[2] > 1:
            frame = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
    else:
        frame = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    img_rgb = frame.astype(np.float32) / 255
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l_rs = cv.resize(img_lab.copy(), (w_in, h_in))[:, :, 0]
    inputs[input_blob] = img_l_rs


                                
    # Network Infer

    res = exec_net.infer(inputs=inputs)

    # Postprocessing 
    update_res = np.squeeze(res[output_blob])

    out = update_res.transpose((1, 2, 0))
    out = cv.resize(out, (w_orig, h_orig))
    img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

    original_image = cv.resize(img, imshow_size)
    grayscale_image = cv.resize(frame, imshow_size)
    colorize_image = (cv.resize(img_bgr_out, imshow_size) * 255).astype(np.uint8)
    lab_image = cv.resize(img_lab_out, imshow_size).astype(np.uint8)

    original_image = cv.putText(original_image, 'Original', (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    grayscale_image = cv.putText(grayscale_image, 'Grayscale', (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    colorize_image = cv.putText(colorize_image, 'Colorize', (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    lab_image = cv.putText(lab_image, 'LAB interpretation', (25, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    #ir_image = [cv.hconcat([original_image, grayscale_image]),
                #cv.hconcat([lab_image, colorize_image])]
    #final_image = cv.vconcat(ir_image) 

    cv.waitKey()
    cv.destroyAllWindows()

    return colorize_image 

def process_single_image_super_resolution_1032(source, net, exec_net, ie=None, model=None, d=None):
    k = 4
    if not exec_net:
        net = ie.read_network(model=model)
        net.reshape({"0":(1, 3, source.shape[0], source.shape[1]), "1":(1, 3, source.shape[0]*k, source.shape[1]*k)})
        exec_net = ie.load_network(network=net, device_name=d)
    out_blob = next(iter(net.outputs))
    in1 = [source.transpose((2, 0, 1))]
    in2_cubic = cv.resize(source, (source.shape[1] * k, source.shape[0] * k),interpolation=cv.INTER_CUBIC)
    in2 = [in2_cubic.transpose((2, 0, 1))]
    # res
    output = exec_net.infer(inputs = {'0': in1, "1": in2})
    output = output[out_blob][0]
    output = output.transpose(1,2,0)
    return output*255,in2_cubic, net, exec_net


def process_single_image_super_resolution_1033(source, net, exec_net, ie=None, model=None, d=None):
    k = 3
    if not exec_net:
        net = ie.read_network(model=model)
        net.reshape({"0":(1, 3, source.shape[0], source.shape[1]), "1":(1, 3, source.shape[0]*k, source.shape[1]*k)})
        exec_net = ie.load_network(network=net, device_name=d)
    out_blob = next(iter(net.outputs))
    in1 = [source.transpose((2, 0, 1))]
    in2_cubic = cv.resize(source, (source.shape[1] * k, source.shape[0] * k),interpolation=cv.INTER_CUBIC)
    in2 = [in2_cubic.transpose((2, 0, 1))]
    # res
    output = exec_net.infer(inputs = {'0': in1, "1": in2})
    output = output[out_blob][0]
    output = output.transpose(1,2,0)
    return output*255,in2_cubic, net, exec_net

def process_pytorch_example():
    pass

def process_RCAN(source, net, exec_net, ie=None, model=None, d=None):
    k = 2
    if not exec_net:
        net = ie.read_network(model=model)
        # while True:
        #     input = next(iter(net.inputs))
        #     print(net[input])

        # print(net.inputs['input'].shape)
        # net.reshape({"input":(1, 3, source.shape[0], source.shape[1])})
        exec_net = ie.load_network(network=net, device_name=d)
    out_blob = next(iter(net.outputs))
    in1 = cv.resize(source, (640,360), interpolation=cv.INTER_LINEAR)
    in1 = [in1.transpose((2, 0, 1))]
    in2_cubic = None
    # in2_cubic = cv.resize(source, (source.shape[1] * k, source.shape[0] * k),interpolation=cv.INTER_CUBIC)
    # in2 = [in2_cubic.transpose((2, 0, 1))]
    # res
    output = exec_net.infer(inputs = {'input': in1})
    output = output[out_blob][0]
    output = output.transpose(1,2,0)
    return output,in2_cubic, net, exec_net

def small_WDSR_x2():
    pass

def small_WDSR_x3():
    pass

def small_WDSR_x4():
    pass

def large_WDSR_x2():
    pass

def large_WDSR_x3():
    pass

def large_WDSR_x4():
    pass

def resizeImages():
    inputs = glob("images" + "/*")
    for imgname in inputs:
        img = cv.imread(imgname)
        img = cv.resize(img,(256,256),interpolation=cv.INTER_NEAREST)
        # cv.imshow(imgname, img)
        cv.imwrite("converted/"+imgname[imgname.rfind("\\")+1:],img)

# resizeImages()

# можно отсортировать изображения так, чтобы было минимальное количество изменений разрешений
def main():
    args = build_argparser().parse_args()
    #print(args.input)

    # Setup networks
    ie = IECore()
    net_1 = ie.read_network(model = args.model_1)
    net_1.batch_size = 1

    # Load network to device
    exec_net_1 = ie.load_network(net_1, args.device)
    

    process = process_single_image_super_resolution_1033

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)

    inputs = glob(args.input + "/*")
    ie = IECore()

    net = None
    exec_net = None
    

    iter = 0
    prevRes = [0,0]
    for i in inputs:
        iter +=1
        img = cv.imread(i)
        res,cubic = None,None
        if prevRes[0] != np.shape(img)[0] or prevRes[1] != np.shape(img)[1]:
            # print("new resolution")
            prevRes[0] = np.shape(img)[0]
            prevRes[1] = np.shape(img)[1]
            res,cubic, net,exec_net = process(img, None, None, ie, args.model_2, args.device)
        else:    
            res,cubic, net,exec_net = process(img, net, exec_net, ie, args.model_2, args.device)
        # resaults.add([res,cubic]])

        try:
            os.makedirs(args.output + "\\super_res")
        except:
            pass
        try:
            os.makedirs(args.output + "\\color_super_res")
        except:
            pass

        print(i, "done_superres")
        cv.imwrite(args.output + "\\super_res\\" + i[i.rfind("\\")+1:],res)
        img = color_image(res, net_1, exec_net_1)
        print(i, "done_colorization")
        cv.imwrite(args.output + "\\color_super_res\\" + i[i.rfind("\\")+1:],img)
       
        # cv.imshow("cubic interpolation " + str(iter), cubic)
        # cv.imshow("super resolution " + str(iter), res)
    cv.waitKey(0) 
    cv.destroyAllWindows()

    return

if __name__ == '__main__':
    sys.exit(main() or 0)
