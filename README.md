# Need For Speed Most Wanted Self Driving using Convolution Neural Net


<h3><i>Work in Progress...</i></h3>

Making an AI which can play Need for Speed Most Wanted only by looking at the game, just like humans learn.

## Detecting Lanes
![Edge Detection](/images/nfsmw1.gif)


## Data Gathering

Captured gameplay using python script found in `grabscreen.py`. Run `main.py` to capture screen while playing the game. 
I collected 200,000 images with their corresponding labels i.e keys `w`, `a`, `d`, `wa`, `wd`.


## Training
The model was first trained using AlexNet model. Later found out `InceptionResNetV2` performs better and goes deeper to find the artifacts. I trained it on 200,000 images.
In total it took around 18 hours of training on a local Nvidia RTX 4070 Ti Super (16GB).
**Tensorboard Logs**
![Training](/images/training.png)

## Things left to add:
1. Joystick controls to improve accuracy.
2. Replace AlexNet Model with InceptionResNetV2 Model.
3. Capturing more data, ideally 1 million images, the model showed signs of improvement until the end.
4. Write a bot to generate training data as it is tedious to play the same route over and over again.
5. Give the model all 3 color channel input to improve accuracy.
6. Reinforcement Learning. 


## Installation

To install the required dependencies, you can use `pip` with the provided `requirements.txt` file.

1. Clone the repository:

   ```bash
   git clone https://github.com/strcoder4007/Need-For-Speed-Self-driving.git
   cd Need-For-Speed-Self-driving
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This command will install all necessary packages including `opencv-python`, `pypiwin32`, and `tensorflow`.

## Pre-trained Model

Checkpoint for a trained NFS Most Wanted: https://drive.google.com/file/d/1RRwhSMUrpBBRyAsfHLPGt1rlYFoiuus2/view?usp=sharing


## Usage

1. To capture screen and gather data for training:
   ```bash
   python main.py
   ```

2. To train the model run:
   ```bash
   python train_model.py
   ```
   
3. To do inference on the model. Run the game and then run the following:
   ```bash
   python test_model.py
   ```

4. Monitor training progress and visualize results using Tensorboard visualizations. Logs are stored in the `board` folder. Use it by running:

    ```bash
    tensorboard --logdir "log"
    ```