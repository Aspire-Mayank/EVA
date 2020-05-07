#  SelfDrivingCarEnv
Simple auto driving car using T3D Reinforcement learning

# Steps to Run
* Because of unavailability of GPU on desktop, Trained it on Google Colab. But running Kivy on colab is very complex, hence once training is done on Colab, model is downloaded to pytorch_models directory and then it is run on my local desktop.
* Use `SelfDrivingCarEnv.ipynb` to train on Colab
* For inference, run: `python Carkv.py`

# Implementation Approch 
* Because we can't run Kivy on Colab and because we don't have GPUs on our desktop, following is the strategy to train
  * Load the sand image and move the coordinates (Vector) in env.Step() for simulating car movement.
  * State is captured by cropping a portion of sand image from car's position. And then rotating it in the direction of the car in such a way that car's orientation is horizontal i.e 0 degrees from x-axis. This state is passed to Actor network
  * Action is 1 dimensional, with its value being amount of angle the car should rotate

# Network Architecture
* We build one neural network for the Actor model and one neural network for the Actor target
* We tried to creat basic Network, transformation failed and second attempt used MobileNet for this implementation as its simple with good speed.

# Understand environment for Kivy Simulation in colab
* We simulate Kivy environment here. As Kivy doesn't do much apart from Graphics
* We maintain x,y position and car's angle. This is rotated based on action
* Action here is one-dimensional, which is the amount of degrees the car should rotate
* If x,y position is on sand, we set a small velocity, else a slightly high velocity
* Our state here corresponds to the cropped portion of current postion as center. This image is rotated to be in the direction of car. This was our network understands car's orientation
* Cropping here is done differently. If we directly crop and rotate the image, we may loose information from the edges. Hence we do the following:
  ![ImageCropProccess](images/croppedImage.png)
  * Crop a larger portion of image
  * Rotate it to make the cropped image in the direction of car's orientation
  * Then crop it to required size
