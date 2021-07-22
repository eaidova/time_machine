from models.single_image_super_resolution import single_image_super_resolution
from models.colorization_siggraph import colorization_siggraph
from models.colorization_v2 import colorization_v2
from models.colorization_v2_old import colorization_v2_old
from models.deoldify import deoldify
from models.deoldify_old import deoldify_old
from models.scratch_detection import scratch_detection
from models.scratch_repair import scratch_repair
# from models.wdsr import wdsr # broken
from models.rcan import rcan
from models.sr import sr
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2 as cv
from threading import Thread, Lock
import time

sys.path.append("C:\Program Files (x86)\IntelSWTools\openvino_2021.4.582\inference_engine\demos\common\python")
from images_capture import open_images_capture

def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified

mutex = Lock()
in1 = []
out1_in2 = []
out2_in3 = []
out3_in4 = []
out4 = []

vw = None
image_iter = 0
def callback(res):
    global image_iter,vw
    if image_iter > 3000:
        if vw:
            vw.release()
            print("released video")
            vw=None
        return

    if not vw:
        if type(res) == list:
            vw = cv.VideoWriter("out.avi",cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),20, (res[0].shape[1], res[0].shape[0]))
        else:
            vw = cv.VideoWriter("out.avi",cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),20, (res.shape[1], res.shape[0]))

    if type(res) == list:
        for i in res:
            cv.imwrite("output\\"+str(image_iter)+".jpg",i)
            vw.write(i) 
            image_iter += 1  
    else:
        cv.imwrite("output\\"+str(image_iter)+".jpg",res)
        vw.write(res)
        print("writed frame")
        image_iter += 1

    mutex.acquire()
    if len(out4):
        out4.pop(0)
    mutex.release()

def input_from_video(path, input_list, net):
    # input_list.append(cv.imread("in.png"))
    cap = open_images_capture(path, True)
    while True:
        while len(in1) < net.max_requests:
            frame = cap.read()
            # frame = cv.resize(frame,None,fx=0.5,fy=0.5)
            input_list.append(frame)
            print("imported frame", len(in1),len(out1_in2),len(out2_in3),len(out3_in4))
        time.sleep(0.1)

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                level=log.INFO, stream=sys.stdout)

    ie = IECore()
    device = "CPU"
    max_requests = 3

    net1 = scratch_detection(
        ie, 
        "scratch_models\\scratch_detector.xml", 
        device,
        get_plugin_configs('GPU', 0, 8),
        in1,
        out1_in2,
        max_requests
    )

    net2 = scratch_repair(
        ie, 
        "scratch_models\\gmcnn-places2-tf.xml", 
        device,
        get_plugin_configs('GPU', 0, 8),
        out1_in2,
        out2_in3,
        max_requests
    )

    net3 = colorization_siggraph(
        ie, 
        "color_models\\colorization-siggraph\\FP16\\colorization-siggraph.xml", 
        device,
        get_plugin_configs('GPU', 0, 8),
        out2_in3,
        out3_in4,
        max_requests
    )

    # net3 = colorization_v2(
    #     ie, 
    #     "color_models\\colorization_v2\\FP16\\colorization-v2.xml", 
    #     device,
    #     get_plugin_configs(device, 0, 8),
    #     out2_in3,
    #     out3_in4,
    #     max_requests
    # )

    # net3 = colorization_v2_old(
    #     ie, 
    #     "color_models\\colorization_v2_old\\FP16\\colorization-v2.xml", 
    #     device,
    #     get_plugin_configs(device, 0, 8),
    #     out2_in3,
    #     out3_in4,
    #     max_requests,
    #     None,
    #     "color_models\\colorization_v2_old\\colorization-v2.npy", 
    # )

    # net3 = deoldify(
    #     ie, 
    #     # "color_models\\deoldify_stable\\deoldify_stable.xml", 
    #     # "color_models\\deoldify_art\\deoldify_art.xml", 
    #     "color_models\\deoldify_video\\deoldify_video.xml", 
    #     device,
    #     get_plugin_configs(device, 0, 8),
    #     out2_in3,
    #     out3_in4,
    #     max_requests
    # )

    # net3 = deoldify_old(
    #     ie, 
    #     "color_models\\deoldify_old\\deoldify_old.xml", 
    #     device,
    #     get_plugin_configs(device, 0, 8),
    #     out2_in3,
    #     out3_in4,
    #     max_requests
    # )

    net4 = single_image_super_resolution(
        ie, 
        "super_res_models\\intel\\single-image-super-resolution-1033\\FP16-INT8\\single-image-super-resolution-1033.xml", 
        # "super_res_models\\intel\\single-image-super-resolution-1032\\FP16-INT8\\single-image-super-resolution-1032.xml", 
        'GPU',
        get_plugin_configs('GPU', 0, 8),
        out3_in4,
        out4,
        max_requests,
        callback,
        # True
    )

    # net4 = rcan(
    #     ie,  
    #     "super_res_models\\rcan\\FP16\\rcan_x4.xml", 
    #     device,
    #     get_plugin_configs(device, 0, 8),
    #     out3_in4,
    #     out4,
    #     max_requests,
    #     callback
    # )

    # net4 = sr(
    #     ie,  
    #     # "super_res_models\\srgan_x4\\saved_model.xml",
    #     "super_res_models\\edsr_x4\\saved_model.xml",
    #     device,
    #     get_plugin_configs(device, 0, 8),
    #     out3_in4,
    #     out4,
    #     max_requests,
    #     callback
    # )

    t = Thread(target=input_from_video,args=["input.mp4", in1, net1])
    t.start()


    
    

if __name__ == '__main__':
    sys.exit(main())

