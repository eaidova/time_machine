from models.model import Model
import numpy as np
import cv2 as cv

class colorization_v2_old(Model):
    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, collback = None, npy_path=None):
        super().__init__(ie, model_path, device, config, input_list, output_list, max_requests, collback)

        self.npy_path = npy_path

        self.input_blob = next(iter(self.net.input_info))
        self.input_shape = self.net.input_info[self.input_blob].input_data.shape

        self.inputs = {}
        for input_name in self.net.input_info:
            self.inputs[input_name] = np.zeros(self.net.input_info[input_name].input_data.shape)
        
        self.output_blob = next(iter(self.net.outputs))

        _, _, self.h_in, self.w_in = self.input_shape
        self.color_coeff = np.load(npy_path).astype(np.float32)

    def process_image(self,img):

        # Reading the frame
        (h_orig, w_orig) = img.shape[:2]
        imshow_size = (w_orig, h_orig )
                
        # Prepare frame

        if img.shape[2] > 1:
                frame = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
        else:
            frame = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        img_rgb = frame.astype(np.float32) / 255
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
        img_l_rs = cv.resize(img_lab.copy(), (self.w_in, self.h_in))[:, :, 0]
        self.inputs[self.input_blob] = img_l_rs

                                    
        # Network Infer

        res = self.exec_net.infer(inputs=self.inputs)

        # Postprocessing 
        update_res = (res[self.output_blob] * self.color_coeff.transpose()[:, :, np.newaxis, np.newaxis]).sum(1)
    

        out = update_res.transpose((1, 2, 0))
        out = cv.resize(out, (w_orig, h_orig))
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

        return (cv.resize(img_bgr_out, imshow_size) * 255).astype(np.uint8)
