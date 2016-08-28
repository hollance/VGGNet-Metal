# VGGNet+Metal

This is the source code that accompanies my blog post [Convolutional neural networks on the iPhone with VGGNet](http://TODO).

This project shows how to implement the 16-layer VGGNet convolutional neural network to do image recognition on the iPhone.

VGGNet was a competitor in the [ImageNet ILSVRC-2014](http://image-net.org/challenges/LSVRC/2014/results) image classification competition and scored second place. The iPhone app uses the version from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). For more details about VGGNet, see the [project page](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) and the [paper](http://arxiv.org/pdf/1409.1556):

    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556

You need an iPhone that supports Metal (and Metal Performance Shaders) to run this app.

**NOTE:** The source code won't run as-is. You need to do the following before you can build the Xcode project:

1 - Download the [prototxt](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt) file.

2 - Download the [caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) file.

3 - Run the conversion script from Terminal (requires Python 3 and the numpy and google.protobuf packages):

```
$ python3 convert_vggnet.py VGG_ILSVRC_16_layers_deploy.prototxt VGG_ILSVRC_16_layers.caffemodel ./output
```

This generates the file `./output/parameters.data`. It will take a few minutes! The reason you need to download the caffemodel file and convert it yourself is that `parameters.data` is a 500+ MB file and you can't put those on GitHub.

4 - Copy `parameters.data` into the `VGGNet-iOS/VGGNet` folder. 

5 - Now you can build the app in Xcode. You can only build for the device, the simulator isn't supported (gives compiler errors).

The VGGNet+Metal source code is licensed under the terms of the MIT license.
