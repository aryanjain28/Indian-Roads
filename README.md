# Introduction
An android application that detects objects usually found on an indian road.
Images for the specific categories is downloaded from COCO dataset, using a script that I wrote (available in Script folder in repository).
Categories includes 8 objects, they are : 
    
      1. Cow
      2. Dog
      3. Truck
      4. Bicycle
      5. Truck
      6. MotorCycle
      7. Person
      8. Bus
      
I trained, model from "ssd_mobilenet_v2_quantized_300x300_coco" tensorflow model's zoo and converted it to TF-Lite in order to use it in mobile.
    
# Usage

The application simply uses phones rear camera and detects objects in real time

# Screenshots

<br>
<p align="center"><img width="460" src="https://raw.githubusercontent.com/aryanjain28/Indian-Roads/master/Screenshot.mp4"></p>
<br>

# References

<a herf="https://towardsdatascience.com/detecting-pikachu-on-android-using-tensorflow-object-detection-15464c7a60cd">https://medium.com</a><br>
<a herf="https://cocodataset.org/">https://cocodataset.org/</a><br>
<a href="https://www.tensorflow.org/lite/" >https://www.tensorflow.org/lite/</a><br>


