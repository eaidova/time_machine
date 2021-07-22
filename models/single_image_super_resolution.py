# https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md#image-processing
import cv2 as cv
from models.model import Model

class single_image_super_resolution(Model):
    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, collback = None, model_type=None):
        super().__init__(ie, model_path, device, config, input_list, output_list, max_requests, collback)
        self.image_sizes = [0,0]
        self.scale = 3
        if model_type:
            self.scale = 4
        self.image_sizes_scaled = [0,0]
        self.calc_scaled_sizes()        
        self.output_blob = next(iter(self.net.outputs))

    def calc_scaled_sizes(self):
        self.image_sizes_scaled[0] = self.image_sizes[0] * self.scale
        self.image_sizes_scaled[1] = self.image_sizes[1] * self.scale

    def process_image(self,image):
        image = cv.resize(image,None,fx=0.5,fy=0.5)
        # preprocess
        if image.shape[0] != self.image_sizes[0] or image.shape[1] != self.image_sizes[1]:
            self.image_sizes[0] = image.shape[0]
            self.image_sizes[1] = image.shape[1]
            self.calc_scaled_sizes()
            self.net.reshape({"0":(1, 3, self.image_sizes[0], self.image_sizes[1]), "1":(1, 3, self.image_sizes_scaled[0], self.image_sizes_scaled[1])})
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device,
                                            config=self.config)

        # infer and postprocess
        return self.exec_net.infer(inputs = {'0': [image.transpose((2, 0, 1))], "1": [cv.resize(image, (self.image_sizes_scaled[1], self.image_sizes_scaled[0]), interpolation=cv.INTER_CUBIC).transpose(2,0,1)]})[self.output_blob][0].transpose(1,2,0)*255

