import cv2
import numpy as np
from models.model import Model

class scratch_detection(Model):

    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, collback = None):
        super().__init__(ie, model_path, device, config, input_list, output_list, max_requests, collback)

    def process_image(self,img):
        data_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_data = data_gray.astype(np.float32) - 127.5
        norm_data /= 127.5
        resized_data = cv2.resize(norm_data, (608, 512))
        scratch_input = np.array([[resized_data]])
  
        # Network Infer
        out = self.exec_net.infer(inputs = {'input': scratch_input})

        # Postprocessing 
        mask = np.squeeze(out['output'])
        # mask[mask >= 0.5] = 1.
        # mask[mask < 0.5] = 0.
        
        res_mask = mask.copy()
        h,w, = mask.shape
        for i in range(5,h-5,1):
            for j in range(5,w-5,1):
                if mask[i,j] >= 0.5:
                    for y in range(-5,5,1):
                        for x in range(-5,5,1):
                            res_mask[i+y,j+x]=1
        mask = res_mask
        
        return [img,mask]

