# NNCUDA
## An implementation of neural network using CUDA 

Implementing 4-Layer Neural Network on CUDA. Comparing performance with same architecture implemented using Keras on Tensorflow.

## Dependencies - CUDA

* CMake 3.0.0 or higher (https://cmake.org/download/)

* OpenCV 4.0 or higher (https://opencv.org/releases.html)

* CUDA 9.0 or higher(https://developer.nvidia.com/cuda-downloads)

## Dependencies - Keras

* Tensorflow 1.10.0 or higher (https://www.tensorflow.org/install)

* Keras 2.0.0 or higher (https://github.com/keras-team/keras/releases)

## Test Dataset 

* I use the labelled images from Stanford Dog Datasets. For more details it can be found here http://vision.stanford.edu/aditya86/ImageNetDogs/ .

## Results

The sample output can be found in out.txt;


### Performance 

The performance of the implementation can be inferred from the output logs, i.e. :

    ...
    Batch progress: 1500/1500
    Cost after 0 epoch: 5.66736
    Epoch Training time is: 4.388 sec
    =====================================================

    Batch progress: 1500/1500
    Cost after 1 epoch: 5.72333
    Epoch Training time is: 4.189 sec
    =====================================================

    Batch progress: 1500/1500
    Cost after 2 epoch: 5.87202
    Epoch Training time is: 4.123 sec
    =====================================================


## References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
2. Haykin, S. (1994). Neural networks (Vol. 2). New York: Prentice hall.
3. http://vision.stanford.edu/aditya86/ImageNetDogs/







