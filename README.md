# My First Attempt at a Neural Network (2013)

**Note**: This repository is purely for entertainment and historical purposes. It represents my early attempts at understanding neural networks and computer vision classification, fumbling around in the dark before frameworks like TensorFlow made this accessible. It's a nostalgic look at trying to figure things out when resources were scarce and Medium articles didn't exist to guide the way.

## What Is This?

This is an attempt at implementing a neural network for image classification, specifically designed to work with 28x28 pixel images (common size for MNIST digit dataset) and classify them into 10 categories (digits 0-9).

### Architecture
- Input layer: 784 nodes (28x28 flattened)
- Hidden layer: 784 nodes 
- Output layer: 10 nodes (one per digit)

### Technical Implementation
The code implements several interesting characteristics:
- Custom activation function: `1.0 - (2.0*weight*input)/(weight + input)`
- Noise-based weight updates instead of traditional backpropagation
- Weight clipping mechanism (0 to 1)
- "Best response" tracking system (similar to simulated annealing)

### OpenCL Experiments
The repository also includes `opencl.py`, representing attempts to accelerate neural network computations using OpenCL. While I successfully implemented basic OpenCL operations, creating a fully functional neural network proved challenging. I even reached out to Khronos Group developers (OpenCL maintainers), but the communication gap between machine learning goals and low-level GPU computing proved difficult to bridge.

These OpenCL experiments were eventually made redundant by TensorFlow's CUDA support. While OpenCL showed promise, CUDA's dominance in the ML space and better developer support made it the clear winner for neural network acceleration.

## Modern AI Perspective

Looking back at this code from a modern perspective reveals some fascinating insights. The implementation, while naive, shows interesting intuitions about neural network optimization that parallel more sophisticated approaches:

- The noise-based weight updates resemble concepts found in genetic algorithms and simulated annealing
- The "best response" tracking system implements a basic form of optimization memory
- The custom activation function, while unconventional, shows an attempt to solve the non-linearity problem

### What Modern Approaches Do Differently
Today's neural networks benefit from:
- Automatic differentiation
- Standardized optimizers (Adam, SGD with momentum)
- Batch processing
- Efficient activation functions (ReLU)
- Better initialization strategies
- Proper gradient-based learning

The main limitation of this early attempt was the lack of backpropagation, making learning less efficient. The random noise approach, while creative, is less directed than modern gradient descent methods.

## Historical Context
I didn't achieve any meaningful results until discovering TensorFlow years later. This code represents that early period of experimentation and learning, where understanding neural networks meant trying to build one from scratch with limited resources and knowledge.

## Repository Contents
- Original Python implementation using NumPy
- OpenCL acceleration attempts
- Requires OpenCV and PIL for image processing
- Designed to work with 28x28 pixel images

---
*This repository serves as a time capsule of early machine learning experimentation, preserved for its historical and educational value.*
