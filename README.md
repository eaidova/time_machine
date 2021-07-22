# Time_machine - >This branch is outdated<
Summer Camp 2021 project
# Colorization

Photo models:

Color_colorization_siggraph.py 
Color_colorization_v2.py
Color_colorization_v2_old.py
Color_deoldify.py
Color_deoldify_old.py

How to use:

python <model_name>.py -m <Path to model>.xml -i <Path to image>.<format> !only for colorization_v2_old! -coeffs <Path to .npy file with color coefficients>.npy

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  
  
Video models:

Color_colorization_v2_video.py
Color_deoldify_video.py
Color_deoldify_old_video.py

How to use:

python <model_name>.py -m <Path to model>.xml 
  
!you should put the video with the name "video.mp4" in the directory with the model
