import time
from threading import Thread, Lock
import cv2 as cv

mutex = Lock()

class Model():
    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, callback=None):
        self.device = device
        self.config = config
        self.ie = ie
        self.net = ie.read_network(model_path)
        self.callback = callback
        self.output_list = output_list
        self.max_requests = max_requests
        self.input_list = input_list
        self.exec_net = ie.load_network(network=self.net, device_name=self.device, config=self.config)

        t = Thread(target=self.think)
        t.start()

    def think(self):
        while True:
            while len(self.input_list) and len(self.output_list) < self.max_requests:
                mutex.acquire()
                data = self.input_list.pop(0)
                mutex.release()
                # start = time.time()
                res = self.process_image(data)
                # print("infer time", time.time() - start)
                self.output_list.append(res)
                # self.calc_psnr(data,res)
                if self.callback:
                    self.callback(res)
            time.sleep(0.1)

    def calc_psnr(self,img,res):
        resized = cv.resize(img,None,fx=3,fy=3, interpolation=cv.INTER_CUBIC)
        print(resized.shape, res.shape)
        print(cv.PSNR(res, resized))