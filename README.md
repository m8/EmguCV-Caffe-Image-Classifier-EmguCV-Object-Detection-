# EmguCV-Caffe-Image-Classifier

This code translated from C++(OpenCv) to C#(EmguCV), and it allows to classify 80 images.(https://docs.opencv.org/3.4.0/d5/de7/tutorial_dnn_googlenet.html)

![image](https://user-images.githubusercontent.com/19881231/37555618-64c7af06-29fb-11e8-8570-a427da2a0b79.jpg)

1. Firstly, download GoogLeNet model files: bvlc_googlenet.prototxt and bvlc_googlenet.caffemodel

2. Also you need file with names of ILSVRC2012 classes: synset_words.txt.

Put these files into working directory of this program. Also, you should know that this is not a fully translation of OpenCV code. Some functions can be changed. 

# Explanation
1) Add .prototxt and .caffemodel files to program
```csharp
EgEmgu.CV.Dnn.Importer caffe = Emgu.CV.Dnn.Importer.CreateCaffeImporter("Text.prototxt", "Model.caffemodel");
```
2) Create net
```csharp
Emgu.CV.Dnn.Net net = new Emgu.CV.Dnn.Net()
```
3) Pass the blob
```csharp
 Mat blob = Emgu.CV.Dnn.DnnInvoke.BlobFromImage(resim_Mat, 0.78, size, scalar, true);
 net.SetInput(blob, "data");
```
**This section is under construction.**

# Dependencies
- EmguCV V3.x (http://www.emgu.com/wiki/index.php/Main_Page)
- bvlc_googlenet.prototxt (https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/bvlc_googlenet.prototxt)
- bvlc_googlenet.caffemodel (https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)


# Resources

https://github.com/opencv/opencv/tree/master/samples/dnn
https://docs.opencv.org/3.4.0/d5/de7/tutorial_dnn_googlenet.html
