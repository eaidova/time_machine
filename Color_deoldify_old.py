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
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser

def color_image(input_image_path, model_path):

    # Setup networks
    ie = IECore()
    net_1 = ie.read_network(model = model_path)
    net_1.batch_size = 1
    
    #net_1.reshape({"input":(1,3,512,512)})
    
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

    print(output_shape)

    # Reading the frame
    frame = cv.imread(input_image_path)
    (h_orig, w_orig) = frame.shape[:2]
    imshow_size = (512, 512)
    print(frame.shape)
          
    # Prepare frame
    frame = cv.resize(frame, (w_in, h_in))
    frame = frame.transpose((2, 0, 1)).astype(np.float32)
    inputs[input_blob] = frame
  
                                
    # Network Infer

    res = exec_net.infer(inputs=inputs)

    # Postprocessing 
    update_res = np.squeeze(res[output_blob])
    img_out = update_res.transpose(1,2,0)
    colorize_image = cv.resize(img_out, imshow_size).astype(np.uint8)
    img_bgr_out = cv.cvtColor(colorize_image, cv.COLOR_BGR2RGB)

    cv.imwrite('deoldify_old.png',  img_bgr_out )
    cv.imshow("img",img_bgr_out )

    cv.waitKey()
    cv.destroyAllWindows()

    return

def main():
    args = build_argparser().parse_args()
    print(args.input)
    color_image(args.input, args.model_1)

    return 0

if __name__ == '__main__':
    sys.exit(main() or 0)