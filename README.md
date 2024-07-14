# Need For Speed Most Wanted Self Driving using Convolution Neural Net


<h3><i>Work in Progress...</i></h3>

Making an AI which can play Need for Speed Most Wanted only by looking at the game, just like humans learn.

![Edge Detection](/images/nfsmw1.gif)

## Data Gathering
Captured gameplay using python script found in `grabscreen.py`. Run `main.py` to capture screen while playing the game. 
I collected 200,000 images with their corresponding labels i.e keys `w`, `a`, `d`, `wa`, `wd`.

## Training
The model was first trained using AlexNet model. Later found out `InceptionResNetV2` performs better and goes deeper to find the artifacts. I trained it on 80,000 manually generated images.
In total it took around 2 hours of training on a local Nvidia RTX 4070 Ti Super (16GB).

## Pre-trained Model
Trained NFS Most Wanted model: https://drive.google.com/file/d/1w0dz0DuLSFUyZmeK18SSDf32u70LbE_M/view?usp=sharing
**Tensorboard Logs**
![Training](/images/training.png)

## Things left to add:
- [ ] Joystick controls to improve accuracy.
- [x] Replace AlexNet Model with InceptionResNetV2 Model.
- [ ] Capturing more data, ideally 1 million images, the model showed signs of improvement until the end.
- [ ] Write a bot to generate training data as it is tedious to play the same route over and over again.
- [x] Give the model all 3 color channel input to improve accuracy.
- [ ] Reinforcement Learning. 

## Installation
Code and software only runs on Windows, and is developed on Windows 11. Although after gathering data on Windows we can train on Linux, but the game runs natively on Windows, as do gamepads and virtual gamepads.
1. Install Need for Speed Most Wanted (2005)
2. Install Patch v1.3 for Need for Speed Most Wanted (2005)
3. Install [NFSMW Extra Options](https://github.com/ExOptsTeam/NFSMWExOpts/releases) (v10.0.0.1338)
4. Copy configuration for [NFSMW Extra Options](https://github.com/ExOptsTeam/NFSMWExOpts/releases) (`extra/NFSMWExtraOptionsSettings.ini`) to `C:\Program Files (x86)\EA GAMES\Need for Speed Most Wanted\scripts` 

## Reading Data from the Game (Gather training data)
**start the game:**
1. Start NFS:MW
2. Quick Race -> Custom Race -> Circuit
3. Heritage Heights (or any other)
4. Laps: 1 or more. (NFSMW ExtraOps required to have more than 6 laps)
5. Traffic Level: None
6. Opponents: 0
7. Difficulty: Any (irrelevant)
8. Catch Up: Any (irrelevant)
9. Transmission: Automatic (Manual does not give any advantage in the game)
10. Wait until the race has started and the 3-2-1 GO! has passed

**How to use this repo**
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

3. To capture screen and gather data for training, first start the game and run `main.py`:
   ```bash
   python main.py
   ```
   The `.npy` data file will be generated and saved in the root of the folder

4. Using the training data to train the model run:
   ```bash
   python train_model.py
   ```
   The model weights file `.h5` will be generated in the root of the folder
   
6. To do inference on the model. Run the game and then run the following:
   ```bash
   python test_model.py
   ```

7. Monitor training progress and visualize results using Tensorboard visualizations. Logs are stored in the `board` folder. Use it by running:
    ```bash
    tensorboard --logdir "log"

    
    ```