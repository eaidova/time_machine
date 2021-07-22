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

*FPS is specified using an Intel core i5-8300h processor*


# How to use

Download models 
>https://disk.yandex.ru/d/ZYie6Q8k4Mf8LQ




!!!you need to create images directory with your images!!!

It's how to start script -

Color_superres.py -i images -m1 colorization-v2.xml -m2 single-image-super-resolution-1033.xml

How to use:
Setup Openvino environment: "C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"
