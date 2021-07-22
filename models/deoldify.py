from models.model import Model
import numpy as np
import cv2 as cv

class deoldify(Model):
    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, collback = None):
        super().__init__(ie, model_path, device, config, input_list, output_list, max_requests, collback)
        
        self.net.reshape({"input":(1,3,512,512)})
        self.exec_net = ie.load_network(self.net, device_name=device, config=config)

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
        (h_orig, w_orig) = frame.shape[:2]
        imshow_size = (w_orig, h_orig)
        #print(frame.shape)
            
        # Prepare frame
        np_data = cv.resize(frame,(self.w_in, self.h_in)).astype(np.float32) / 255
        #print(np_data.shape)
        np_data -= np.array([0.485, 0.456, 0.406])
        np_data /= np.array([0.229, 0.224, 0.225])
        input_data = np_data.transpose((2, 0, 1)).astype(np.float32)
        self.inputs[self.input_blob] = input_data
    
                                    
        # Network Infer

        res = self.exec_net.infer(inputs=self.inputs)

        # Postprocessing 
        update_res = np.squeeze(res[self.output_blob])
        img_out = (cv.normalize(update_res, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        return cv.resize(img_out, imshow_size)

