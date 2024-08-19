
# Real-time Facial Expression Recognition.

Welcome to my GitHub repository for the "Real-time Facial Expression Recognition." project! This project is a hands-on exploration of facial recognition technology, employing convolutional neural networks (CNN) implemented in Keras to swiftly and accurately identify and classify facial expressions in real-time. The dataset used comprises 48x48 pixel grayscale images of faces, each associated with one of seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). The CNN is trained to discern distinctive facial features and patterns crucial for emotion classification. OpenCV is integrated for automatic face detection in images, enabling the drawing of bounding boxes around recognized faces. Once trained, the CNN serves model predictions directly to a web interface, facilitating real-time facial expression recognition.

![architecture ](https://github.com/irtika98/Realtime-Facial-expression-recognition/blob/master/arch.png)



## Dataset

The dataset comprises 48x48 pixel black-and-white images of faces, each expressing one of seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). Created by Pierre-Luc Carrier and Aaron Courville for the 2013 Facial Expression Recognition Challenge, the dataset includes 28,708 training images, 7,178 test images. 
# Methodology

## 1. Requirements

The Requirements for this project are given in the requirements.txt file.

```bash
    pip install -r requirements.txt

```


## 2. CNN Architecture Summary

### Model Overview

The Convolutional Neural Network (CNN) is designed for facial expression recognition. The model consists of multiple convolutional layers followed by batch normalization, activation functions (ReLU), max-pooling, dropout layers, and fully connected layers.

## Layers

### a) Convolutional Layers

- **Layer 1:**
  - Filters: 64
  - Kernel Size: (3, 3)
  - Activation: ReLU
  - Batch Normalization
  - Max Pooling: (2, 2)
  - Dropout: 0.25

- **Layer 2:**
  - Filters: 128
  - Kernel Size: (5, 5)
  - Activation: ReLU
  - Batch Normalization
  - Max Pooling: (2, 2)
  - Dropout: 0.25

- **Layer 3:**
  - Filters: 512
  - Kernel Size: (3, 3)
  - Activation: ReLU
  - Batch Normalization
  - Max Pooling: (2, 2)
  - Dropout: 0.25

- **Layer 4:**
  - Filters: 512
  - Kernel Size: (3, 3)
  - Activation: ReLU
  - Batch Normalization
  - Max Pooling: (2, 2)
  - Dropout: 0.25

### b) Flattening Layer

- Flattens the output from the convolutional layers.

### c) Fully Connected Layers

- **Layer 5:**
  - Neurons: 256
  - Activation: ReLU
  - Batch Normalization
  - Dropout: 0.25

- **Layer 6:**
  - Neurons: 512
  - Activation: ReLU
  - Batch Normalization
  - Dropout: 0.25

- **Output Layer:**
  - Neurons: 7 (for 7 emotion categories)
  - Activation: Softmax

## Model Compilation

- **Optimizer:**
  - Adam with learning rate 0.0005

- **Loss Function:**
  - Categorical Crossentropy

- **Metrics:**
  - Accuracy


![summary](https://i.ibb.co/VtCHBCS/summary.png)


## Training and Evaluation

We begin by setting the number of epochs to 15, a crucial parameter in gradient descent that determines how many times the entire training dataset is traversed. To calculate the steps per epoch, we divide the number of images in the training generator by the batch size. This calculation is also performed for the validation set.
Subsequently, we incorporate three essential callbacks during training. The first callback, ReduceLROnPlateau, dynamically adjusts the learning rate when there is no improvement in validation loss for two consecutive epochs. The second callback, ModelCheckpoint, serves to save the model weights with the highest validation accuracy, storing them in HDF5 format—a grid format optimal for storing multi-dimensional arrays.
To monitor the training progress in real time, we employ the PlotLossesKerasTF callback. This allows us to observe both the training loss and accuracy plots per epoch.The initial epoch tends to take longer as it involves GPU resource allocation, loading various libraries, and optimizing files. 


**Accuracy vs epoch plot**
![Accuracy](https://i.ibb.co/6DL3bjg/accuracy.png)

**Loss vs epoch plot**
![Loss](https://i.ibb.co/Sy2Zchn/loss.png)

**metrics summary**
![summary](https://i.ibb.co/JpsfrS9/train-sum.png)


## Saving Model architecture as JSON string
In neural network models, the model's architecture, which outlines the arrangement and connections of its layers, is often serialized to the JSON format. JSON is employed to capture the structure of the model using the to_json() method. 
Following serialization, the obtained JSON string is written to a local file named "model.json," utilizing standard file. This file serves as a configuration document, preserving the model's architecture. It facilitates the recreation and utilization of the same model structure in the future, eliminating the need for retraining.

## Deployment to web
The Flask app is designed to smoothly provide model predictions directly to a web interface. An HTML template forms the basic structure of the Flask application. Within the camera class, the webcam's image stream is utilized. OpenCV is employed to detect faces, outline them with bounding boxes, and convert the images from color to grayscale. Additionally, the images are resized to a standard 48x48 dimension, preparing them for input into the model. This Flask app acts as an accessible platform for obtaining instant predictions from the trained model

## Run 
To make the model work, just run the main.py script. This sets up a Flask app that shows the model's predictions on a website. The camera class handles things by sending images from a webcam to the pre-trained CNN model. The model then figures out what's in the images and labels the video frames. Finally, the labeled images go back to the website. You can use the model on saved videos or in real-time with a webcam.

```bash
    python main.py

```


## Testing model 

**happy expression detected on live web cam**
![happy](https://i.ibb.co/BZDnVnT/happy2.png)

**neutral facial expression detected on live webcam**
![neutral](https://i.ibb.co/k8zzB6L/NEUTRAL2.png)



**happy expression detected on video data**

![happy](https://i.ibb.co/kJT8QhM/happy3.png)

**sad facial expression detected on video data**

![sad](https://i.ibb.co/LtWyQrj/sad.png)

**angry expression detected on video data**

![angry](https://i.ibb.co/qyKgRZt/angry.png)

**surprise facial expression detected on video data**

![suprise](https://i.ibb.co/dJJs31h/suprise.png)


**Disgust facial expression detected on video data**

![Disgust](https://i.ibb.co/TkRDdSZ/disgust.png)

### *To use webcam, open camera.py and change line 11 to:*
```bash
    self.video = cv2.VideoCapture(0)

```
