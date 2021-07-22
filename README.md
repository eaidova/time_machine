# «Time machine» 
This is our **CV Summer Camp 2021 project**

Our program `colorize` old grayscale photos and `improve` their `quality`
***

![Time Machine header](https://user-images.githubusercontent.com/58187114/125828115-b1d74ae2-c2d8-458c-afbd-cdddf95b8874.jpg)
# How it's work
Our program first finds image defects using the `scretch detector network`, then eliminates them using the `inpainting network`, after which the image is colorized using one of `colorization networks` and then image resolution is increased by using one of `superres networks`

![image](https://user-images.githubusercontent.com/58187114/126698343-892f1f6e-ae61-49d0-8bd9-168f0473403f.png)
![resized_img ](https://user-images.githubusercontent.com/58187114/126700673-6296ae20-be31-44d0-9d27-6792fbb13fff.png)

![image](https://user-images.githubusercontent.com/58187114/126699152-cbd134bb-21e8-4931-bac3-7de6128add36.png)
![image](https://user-images.githubusercontent.com/58187114/126699176-bdd90b43-5b04-439e-bb81-ae3408b8cf5b.png)

![image](https://user-images.githubusercontent.com/58187114/126699193-756fae61-d514-43b1-bb24-e0e27417ad8b.png)

# Image restoration 
![image](https://user-images.githubusercontent.com/58187114/126701404-885e0525-522a-4dba-9820-dbe8bec331a5.png)
![image](https://user-images.githubusercontent.com/58187114/126701416-49c41ecf-c267-4d17-b25d-cf3e38bc78c5.png)
![image](https://user-images.githubusercontent.com/58187114/126701426-e18fa243-eaa1-44c5-9463-bcc23ef7ac90.png)

For image restoration, we use `scretch_detector`, this is a pre-trained U-net for searching for cracks and then with the help of the resulting mask and the network `gmcnn-places2-tf` we paint over the cracks 

![image](https://user-images.githubusercontent.com/58187114/126701375-ef6325e3-5915-4162-8488-24a5110dffc6.png)

*FPS is specified using an Intel core i5-8300h processor 

***MOS - Mean Opinion Score***

# Colorization
![image](https://user-images.githubusercontent.com/58187114/126702572-a20db2fb-415c-43f9-a695-4d578a9ba7c4.png)

For colorization, you can use various neural networks

![image](https://user-images.githubusercontent.com/58187114/126702890-ba7feecb-a624-417c-b341-b9582f05167e.png)

*FPS is specified using an Intel core i5-8300h processor **FP16 

![image](https://user-images.githubusercontent.com/58187114/126703004-36d0ac64-69b0-4e84-b2f4-090736a710a9.png)
![image](https://user-images.githubusercontent.com/58187114/126703048-b3b4d0e9-3522-404a-b976-d563d4c2e38b.png)
![image](https://user-images.githubusercontent.com/58187114/126703076-219c8c18-ac28-406e-bc3e-271ce25e308c.png)

# Super resolution
![image](https://user-images.githubusercontent.com/58187114/126703539-21d47bfa-fc50-4440-8e48-4d9bd794671a.png)
![image](https://user-images.githubusercontent.com/58187114/126703546-babc103a-80d5-4382-99c5-fdd68122c2f2.png)

For super_resolution, you can use various neural networks

![image](https://user-images.githubusercontent.com/58187114/126703722-1e29961f-ac8a-44a1-9e84-af3c48b71e2d.png)
 
*FPS is specified using an Intel core i5-8300h processor **FP16 

***PSNR - peak signal-to-noise ratio***

![image](https://user-images.githubusercontent.com/58187114/126704030-84c50244-d6de-471e-8abb-ebeec5c38e4b.png)
![image](https://user-images.githubusercontent.com/58187114/126704064-b37cd1ff-6912-4d94-a52e-746d1b839e7e.png)
![image](https://user-images.githubusercontent.com/58187114/126706063-9b2267ee-c530-40ed-9640-9441c9e14a66.png)

# How to use
Install OpenVino 2021.04
>https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html

Clone repository
>git clone https://github.com/eaidova/time_machine

Download models 
>https://disk.yandex.ru/d/ZYie6Q8k4Mf8LQ
Unzip archive into your directory

Create virtual environment 
>mkdir time_machine && cd time_machine
>python -m venv openvinoenv

Activate it
>python 'time-machine\openvinoenv\Scripts\Activate'

Install requirements
>pip install numpy

Start script
>main.py

If you want to apply other neural networks, go to main.py and uncomment what you want to use and comment out what you don't want to use

**Please make sure that you are sending the correct data to log in to the network**

# Links to models

**Photo restoration**
* `scratch_detector` - https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
* `gmcnn-places2-tf` - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/gmcnn-places2-tf

**Colorization**
* `Deoldify ONNX` - https://github.com/KeepGoing2019HaHa/AI-application/tree/master/deoldify/ncnn
* `Deoldify` - https://github.com/jantic/DeOldify
* `Colorization-v2` - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-v2
* `Colorization-v2_old` - https://github.com/richzhang/colorization/tree/caffe
* `Colorization-siggraph` - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-siggraph

**Super resolution**
* `single-image-super-resolution-1033` - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/single-image-super-resolution-1033
* `single-image-super-resolution-1032` - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/single-image-super-resolution-1032
* `EDSR` - https://github.com/krasserm/super-resolution
* `RCAN` - https://github.com/yulunzhang/RCAN
* `SRGAN` - https://github.com/krasserm/super-resolution

*You can find the models converted to ONNX and IR implementation at the link https://disk.yandex.ru/d/ZYie6Q8k4Mf8LQ*
