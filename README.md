# CS231n: Convolutional Neural Networks for Visual Recognition - Assignment Solutions

This repository contains my solutions to the assignments for Stanford's [CS231n](http://cs231n.stanford.edu/) "Convolutional Neural Networks for Visual Recognition" course (Spring 2020).

Stanford's CS231n is one of the best ways to dive into Deep Learning in general, in particular, into Computer Vision. If you plan to excel in another subfield of Deep Learning (say, Natural Language Processing or Reinforcement Learning), we still recommend that you start with CS231n, because it helps build intuition, fundamental understanding and hands-on skills. Beware, the course is very challenging! 

To motivate you to work hard, here are actual applications that you'll implement in A3 - Style Transfer and Class Visualization. 

<p align=center><img src=https://habrastorage.org/webt/ik/ny/o4/iknyo4fnkbokzoavq6nlsuitc6y.png align=center /><img src=https://habrastorage.org/webt/8t/go/qa/8tgoqaoa1vwmiuagfkx0i4nkjmm.png align=center /></p>

For the one on the left, you take a base image and a style image and apply the "style" to the base image (reminds you of Prisma and Artisto, right?). The example on the right is a random image, gradually perturbed in a way that a neural network classifies it more and more confidently as a gorilla. DIY Deep Dream, isn't it? And it's all math under the hood, it's cool to figure out how it all works. You'll get to this understanding with CS231n, it'll be hard but at the same time an exciting journey from a simple kNN implementation to these fascinating applications. If you think that these two applications are eye-catchy, then take another look at the picture above - a Convolutional Neural Network classifying images. That's the basics of how machines can "see" the world. The course will teach you both how to build such an algorithm from scratch and how to use modern tools to run state-of-the-art models for your tasks. 

Find course notes and assignments [here](http://cs231n.github.io) and be sure to check out the video lectures for [Winter 2016](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) and [Spring 2017](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)!

Assignments have been completed using both TensorFlow and PyTorch.

## Assignment #1: Image Classification, kNN, SVM, Softmax, Neural Network
Q1: [k-Nearest Neighbor Classifier](assignment1/knn.ipynb)
- Test accuracy on CIFAR-10: 0.282

Q2: [Training a Support Vector Machine](assignment1/svm.ipynb)
- Test accuracy on CIFAR-10: 0.376

Q3: [Implement a Softmax classifier](assignment1/softmax.ipynb)
- Test accuracy on CIFAR-10: 0.355

Q4: [Two-Layer Neural Network](assignment1/two_layer_net.ipynb)
- Test accuracy on CIFAR-10: 0.501

Q5: [Higher Level Representations: Image Features](assignment1/features.ipynb)
- Test accuracy on CIFAR-10: 0.576

## Assignment #2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets
Q1: [Fully-connected Neural Network](assignment2/FullyConnectedNets.ipynb)
- Validation / test accuracy on CIFAR-10: 0.547 / 0.539

Q2: [Batch Normalization](assignment2/BatchNormalization.ipynb)

Q3: [Dropout](assignment2/Dropout.ipynb)

Q4: [Convolutional Networks](assignment2/ConvolutionalNetworks.ipynb)

Q5: [PyTorch](assignment2/PyTorch.ipynb) / [TensorFlow v2](assignment2/TensorFlow.ipynb) on CIFAR-10 / [TensorFlow v1](assignment2/TensorFlow_v1.ipynb) ([Tweaked TFv1 model](assignment2/TensorFlow_Tweaked_Model_v1.ipynb))
- Training / validation / test accuracy of TF implementation on CIFAR-10: 0.928 / 0.801 / 0.822
- PyTorch implementation:

| Model       | Training Accuracy | Test Accuracy |
| ----------- |:-----------------:| :------------:|
| Base network | 92.86 | 88.90 |
| VGG-16  | 99.98  | 93.16 |
| VGG-19  | 99.98  | 93.24 |
| ResNet-18  | 99.99  | 93.73 |
| ResNet-101  | 99.99 | 93.76 |

## Assignment #3: Image Captioning with Vanilla RNNs, Image Captioning with LSTMs, Network Visualization, Style Transfer, Generative Adversarial Networks
Q1: [Image Captioning with Vanilla RNNs](assignment3/RNN_Captioning.ipynb)

Q2: [Image Captioning with LSTMs](assignment3/LSTM_Captioning.ipynb)

Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images ([PyTorch](assignment3/NetworkVisualization-PyTorch.ipynb) / [TensorFlow v2](assignment3/NetworkVisualization-TensorFlow.ipynb) / [TensorFlow v1](assignment3/NetworkVisualization-TensorFlow_v1.ipynb))

Q4: Style Transfer ([PyTorch](assignment3/StyleTransfer-PyTorch.ipynb) / [TensorFlow v2](assignment3/StyleTransfer-TensorFlow.ipynb) / [TensorFlow v1](assignment3/StyleTransfer-TensorFlow_v1.ipynb))

Q5: Generative Adversarial Networks ([PyTorch](assignment3/Generative_Adversarial_Networks_PyTorch.ipynb) / [TensorFlow v2](assignment3/Generative_Adversarial_Networks_TF.ipynb) / [TensorFlow v1](assignment3/Generative_Adversarial_Networks_TF_v1.ipynb))

## Course notes
- My [course notes](https://github.com/amanchadha/stanford-cs231n-notes-2020/tree/master/notes)
- Official [course notes](https://cs231n.github.io/)
- Reading material that I found to be useful for [Assignment 2](assignment2/Reading%20material) and [Assignment 3](assignment3/Reading%20material)

## GPUs
For some parts of the 3rd assignment, you'll need GPUs. Kaggle Kernels or Google Colaboratory will do.

## Useful links
- The official [course website](http://cs231n.stanford.edu/) 
- Video-lectures. Prerequisites are given in the 1st lecture.
	- Winter 2016 [YouTube playlist](https://www.youtube.com/playlist?list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG)
	- Spring 2017 [YouTube playlist](https://goo.gl/pcj7c8)
- [Syllabus](http://cs231n.stanford.edu/syllabus.html) with assignments

## Direct links to Spring 2017 lectures
- [Lecture 1](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- [Lecture 2](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=2)
- [Lecture 3](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=3) 
- [Lecture 4](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4)
- [Lecture 5](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=5) 
- [Lecture 6](https://www.youtube.com/watch?v=wEoyxE0GP2M&index=6&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- [Lecture 7](https://www.youtube.com/watch?v=_JB0AO7QxSA&index=7&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) 
- [Lecture 8](https://www.youtube.com/watch?v=6SlgtELqOWc&index=8&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- [Lecture 9](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=9) 
- [Lecture 10](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=10)
- [Lecture 11](https://www.youtube.com/watch?v=nDPWywWRIRo&index=11&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) 
- [Lecture 12](https://www.youtube.com/watch?v=6wcs6szJWMY&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=12)
- [Lecture 13](https://www.youtube.com/watch?v=5WoItGTWV54&index=13&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) 
- [Lecture 14](https://www.youtube.com/watch?v=lvoHnicueoE&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=14)
- [Lecture 15](https://www.youtube.com/watch?v=eZdOkDtYMoo) 
- [Lecture 16](https://www.youtube.com/watch?v=CIfsB_EYsVI&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=16)
