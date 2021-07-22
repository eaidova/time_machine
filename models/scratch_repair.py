import cv2
import numpy as np
from models.model import Model

class scratch_repair(Model):

    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, collback = None):
        super().__init__(ie, model_path, device, config, input_list, output_list, max_requests, collback)

    def process_image(self,img_mask):
        img = img_mask[0]
        mask = img_mask[1]
        if mask.max() < 0.5:
            return img

        imshow_size = (img.shape[1], img.shape[0])

        # Prepare frame
        resized_img = cv2.resize(img, (680, 512)).astype(np.float32)
        resized_mask = cv2.resize(mask, (680, 512)).astype(np.float32)
        if len(resized_mask.shape) == 2:
            resized_mask = np.expand_dims(resized_mask, -1)
        resized_mask[resized_mask >= 0.5] = 1.
        resized_mask[resized_mask < 0.5] = 0.
        resized_img = resized_img * (1 - resized_mask) + 255 * resized_mask
        inpainting_input = np.expand_dims(resized_img.transpose(2, 0, 1), 0)
        inpainting_mask = np.expand_dims(resized_mask.transpose(2, 0, 1), 0)
                                    
        # Network Infer
        out = self.exec_net.infer(inputs = {'Placeholder': inpainting_input, 'Placeholder_1': inpainting_mask})

        # Postprocessing
        out_img = out['Minimum'][0].transpose((1, 2, 0)).astype(np.uint8)
        res = cv2.resize(out_img, imshow_size)
        
        cv2.imwrite("asd.png",res)
        
        return res
    