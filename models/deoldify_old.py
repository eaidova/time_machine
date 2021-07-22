from models.model import Model
import numpy as np
import cv2 as cv

class deoldify_old(Model):
    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, collback = None):
        super().__init__(ie, model_path, device, config, input_list, output_list, max_requests, collback)
        self.net.batch_size = 1
        self.input_blob = next(iter(self.net.input_info))
        self.input_shape = self.net.input_info[self.input_blob].input_data.shape

        self.inputs = {}
        for input_name in self.net.input_info:
            self.inputs[input_name] = np.zeros(self.net.input_info[input_name].input_data.shape)
        
        self.output_blob = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_blob].shape

        _, _, self.h_in, self.w_in = self.input_shape

    def process_image(self,frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        img_size = (512, 512)
            
        # Prepare frame
        frame = cv.resize(frame, (self.w_in, self.h_in))
        frame = frame.transpose((2, 0, 1))
        self.inputs[self.input_blob] = frame
    
                                    
        # Network Infer
        res = self.exec_net.infer(inputs=self.inputs)

        # Postprocessing 
        update_res = np.squeeze(res[self.output_blob])
        img_out = update_res.transpose(1,2,0)
        colorize_image = cv.resize(img_out, img_size).astype(np.uint8)
        return cv.cvtColor(colorize_image, cv.COLOR_BGR2RGB)