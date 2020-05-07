#  SelfDrivingCarEnv
Simple auto driving car using T3D Reinforcement learning

## Steps to Run
* Because of unavailability of GPU on desktop, Trained it on Google Colab. But running Kivy on colab is very complex, hence once training is done on Colab, model is downloaded to pytorch_models directory and then it is run on my local desktop.
* Use `SelfDrivingCarT3D.ipynb` to train on Colab
* For inference, run: `python Carkv.py`

## Implementation Approch 
* Because we can't run Kivy on Colab and because we don't have GPUs on our desktop, following is the strategy to train
  * Load the sand image and move the coordinates (Vector) in env.Step() for simulating car movement.
  * State is captured by cropping a portion of sand image from car's position. And then rotating it in the direction of the car in such a way that car's orientation is horizontal i.e 0 degrees from x-axis. This state is passed to Actor network
  * Action is 1 dimensional, with its value being amount of angle the car should rotate

## Network Architecture
* We build one neural network for the Actor model and one neural network for the Actor target
* We tried to creat basic Network, transformation failed and second attempt used MobileNet for this implementation as its simple with good speed.

## Understand environment for Kivy Simulation in colab
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
  

# Self Driving Car with Custom Environment using Kivy on T3D Network
## *Flow Diagram* :
1. Input Values : State_dim -> cropImage
2. Output Values : action_dim -> angle or rotation
3. Max_Action : -> car directional movement respect to goal with some angle 
4. Helper value : -> Distance between goals

### Step *1: Change Actor Critic Model to CNN*
1.  Input value has crop Image state_dim (32,32,1) with action_dim int.
2.  Need to concant post convolution with linear conversion and use of GAP in forward function.

### Step 2: *As part of Custom Env Creation, utility functions creation.*
1.  In given p2s7 files replace T3D env.step() with update() function with modification with Input given angle as action it will return next_obs, reward, done, Info
2.  In custom Enviroment move() replace env.action_space() function with given input action as rotation, it will calculate next obs with help of Crop image and car centroid.
3. Created utitlityImage.py File which gives patch image with help of car centroid in map which can be pass to CNN network to learn direction with help of triangular pixel, to get idea of car movement with respect to Goals.
4. Created reset() which will re-initialize all values. it will happen once episode over. 

### Step 3: Define Multiple Goals and Done Parameters Based on Car location
*Below Points for Done Parameters to reset()*: 
1. As car reached to Wall with padding sand, 'done' will be True.
2. As worst scenerio 1000 steps if car not able to reach target. 
3. car reached to Goal, done will be True.

### Step 4: Define Reward based on goal distance and angle.
1. Define in Map.py. 
2. Need to Draw graph of Penality with angle as action taken. 

### Step 5: Integration T3D Actor-Critic CNN model with Map.py
1. As per input value Patch need changes for CNN.
2. Setup network as per change size or image dimension in network.
3. Re-use Custom Environment Method while intergation.
4. reset(), move(), update(), crop() all can be modify and use while inferencing.

### Step 6: Helper
1. As suggested Divide Problem In Two Part:
>A. Provide Blank Map to Car(agent) without Sand or Road with only goal as coordinate. and see learnings of agent.
>B. Provide only Sand to Car(agent) with no goals, and see how it performing or following road.

### UseFul Clarification :
> *Think on below points*
1. How to get direction of car.
2. how CNN model get direction from image.
3. how to crop image from main image.


# Happy Learning..!
