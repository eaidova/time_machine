import cv2 as cv
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import argparse
import time
import sys

def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m1', dest='model_1', default='deoldify', 
                      help='Path to the model_1')
    return parser

def color_image(model_path):

    output = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1280, 720))

    # Setup networks
    ie = IECore()
    net_1 = ie.read_network(model = model_path)
    net_1.batch_size = 1
    
    net_1.reshape({"input":(1,3,512,512)})
    
    # Load network to device
    exec_net = ie.load_network(net_1, 'CPU')  

    input_blob = next(iter(net_1.input_info))
    input_shape = net_1.input_info[input_blob].input_data.shape

    inputs = {}
    for input_name in net_1.input_info:
        inputs[input_name] = np.zeros(net_1.input_info[input_name].input_data.shape)
    
    output_blob = next(iter(net_1.outputs))
    output_shape = net_1.outputs[output_blob].shape

    _, _, h_in, w_in = input_shape


    cap = cv.VideoCapture('video.mp4')
    ret, img = cap.read() 
    while(cap.isOpened()):

        # Reading the frame
        ret, frame = cap.read()
        (h_orig, w_orig) = frame.shape[:2]
        imshow_size = (w_orig, h_orig)
              
        # Prepare frame
        np_data = cv.resize(frame,(w_in, h_in)).astype(np.float32) / 255
        np_data -= np.array([0.485, 0.456, 0.406])
        np_data /= np.array([0.229, 0.224, 0.225])
        input_data = np_data.transpose((2, 0, 1)).astype(np.float32)
        inputs[input_blob] = input_data
                                    
        # Network Infer

        res = exec_net.infer(inputs=inputs)

        # Postprocessing 
        update_res = np.squeeze(res[output_blob])
        img_out = (cv.normalize(update_res, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        resize_image = cv.resize(img_out, imshow_size)

        
        output.write(cv.resize(  resize_image, (1280, 720)))
        cv.imshow("img",  resize_image)
        print(ret)

        if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
            break

    return

def main():
    args = build_argparser().parse_args()
    color_image(args.model_1)

    return 0

if __name__ == '__main__':
    sys.exit(main() or 0)
