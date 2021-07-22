# Time_machine - >This branch is outdated<
Summer Camp 2021 project
# Scratch_detection
How to use:
Setup Openvino environment: "C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"

scratch_detection: https://drive.google.com/file/d/1jyWva9O5E3DB_7iozgzG_Ld8WYspY5uQ/view?usp=sharing

images_inpainting: https://docs.openvinotoolkit.org/2021.3/omz_models_model_gmcnn_places2_tf.html

usage algorithm
1) upload a photo
2) convert it to grayscale and normalize the data
3) resize the data to the input size of the first network
4) start execution of the scratch_detection network
5) identify areas of defects
6) expand the areas of defects (due to insufficient accuracy of the network)
7) resize the input image and mask to the input size of the second network
8) apply a mask to the image
9) transpose the image and mask to the format (N, C, H, W)
10) start network execution gmcnn_places2_tf
