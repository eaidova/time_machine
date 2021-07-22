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
    args.add_argument('-m', dest='model_1', default='colorization-v2', 
                      help='Path to the model_1')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser

def color_image(input_image_path, model_path):

    # Setup networks
    ie = IECore()
    net_1 = ie.read_network(model = model_path)
    net_1.batch_size = 1

    # Load network to device
    exec_net = ie.load_network(net_1, 'CPU')

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

    


    # Reading the frame
    img = cv.imread(input_image_path)
    (h_orig, w_orig) = img.shape[:2]
    imshow_size = (w_orig, h_orig )
            
    # Prepare frame

    if img.shape[2] > 1:
            frame = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
    else:
        frame = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    img_rgb = frame.astype(np.float32) / 255
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l_rs = cv.resize(img_lab.copy(), (w_in, h_in))[:, :, 0]

    user_ab = np.zeros((2,256,256))
    user_map = np.zeros((1,256,256))
                                
    # Network Infer

    res = exec_net.infer(inputs = {'data_l': img_l_rs, "user_ab": user_ab, "user_map": user_map})

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

    original_image = cv.putText(original_image, 'Original', (25, 50),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    grayscale_image = cv.putText(grayscale_image, 'Grayscale', (25, 50),
                                     cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    #colorize_image = cv.putText(colorize_image, 'Colorize', (25, 50),
                                    #cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    lab_image = cv.putText(lab_image, 'LAB interpretation', (25, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    ir_image = [cv.hconcat([original_image, grayscale_image]),
                cv.hconcat([lab_image, colorize_image])]
    final_image = cv.vconcat(ir_image) 

    cv.imwrite('colorization-siggraph.png', colorize_image)
    cv.imshow("img", final_image)

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
