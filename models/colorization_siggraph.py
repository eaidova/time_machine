import cv2 as cv
from models.model import Model
import numpy as np

class colorization_siggraph(Model):
    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, collback = None):
        super().__init__(ie, model_path, device, config, input_list, output_list, max_requests, collback)

        self.input_blob = next(iter(self.net.input_info))
        self.output_blob = next(iter(self.net.outputs))
        _, _, self.h_in, self.w_in = self.net.input_info[self.input_blob].input_data.shape

    def process_image(self,img):

        # Prepare output shape
        (h_orig, w_orig) = img.shape[:2]
        imshow_size = (w_orig, h_orig)
                
        # Prepare frame
        if img.shape[2] > 1:
                frame = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
        else:
            frame = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        img_rgb = frame.astype(np.float32) / 255
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
        img_l_rs = cv.resize(img_lab.copy(), (self.w_in, self.h_in))[:, :, 0]
        user_ab = np.zeros((2,256,256))
        user_map = np.zeros((1,256,256))
     
        # Network Infer
        res = self.exec_net.infer(inputs = {'data_l': img_l_rs, "user_ab": user_ab, "user_map": user_map})

        # Postprocessing 
        update_res = np.squeeze(res[self.output_blob])

        out = update_res.transpose((1, 2, 0))
        out = cv.resize(out, (w_orig, h_orig))
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

        return (cv.resize(img_bgr_out, imshow_size) * 255).astype(np.uint8)