# https://github.com/krasserm/super-resolution
from models.model import Model

class sr(Model):
    def __init__(self, ie, model_path, device, config, input_list, output_list, max_requests, collback = None):
        super().__init__(ie, model_path, device, config, input_list, output_list, max_requests, collback)
        self.image_sizes = [0,0]
        self.output_blob = next(iter(self.net.outputs))
        self.input_blob = next(iter(self.net.input_info))

    def process_image(self,image):
        # preprocess
        if image.shape[0] != self.image_sizes[0] or image.shape[1] != self.image_sizes[1]:
            self.image_sizes[0] = image.shape[0]
            self.image_sizes[1] = image.shape[1]
            self.net.reshape({self.input_blob:(1, 3, self.image_sizes[0], self.image_sizes[1])})
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device,
                                            config=self.config)

        # infer and postprocess
        return self.exec_net.infer(inputs = {self.input_blob: [image.transpose((2, 0, 1))]})[self.output_blob][0].transpose(1,2,0)

