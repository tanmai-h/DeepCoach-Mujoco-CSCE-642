# Continuous Control for High-Dimensional State Spaces: An Interactive Learning Approach
This is a fork of the original [Continuous Control for High-Dimensional State Spaces: An Interactive Learning Approach](https://github.com/rperezdattari/Continuous-Control-for-High-Dimensional-State-Spaces-An-Interactive-Learning-Approach) project.

The code in this is project is modified as part of an academic procject to test the effectiveness of deep COACH in training MuJoCo Fetch Domains (Pick And Place, Slide and Push).
All code changes on top of the original codebase is authored by: Manas Sahu And Harish Tanmai

## Original README.md
Code of the paper "Continuous Control for High-Dimensional State Spaces: An Interactive Learning Approach" submitted to ICRA 2019.

This repository is an extension of [Interactive Learning with Corrective Feedback for Policies based on Deep Neural Networks](https://github.com/rperezdattari/Interactive-Learning-with-Corrective-Feedback-for-Policies-based-on-Deep-Neural-Networks). The *enhanced* version of D-COACH was added in this project; the version presented in "Interactive Learning with Corrective Feedback for Policies based on Deep Neural Networks" is now called *basic*.

This code is based on the following publication:
1. [Continuous Control for High-Dimensional State Spaces: An Interactive Learning Approach](http://www.jenskober.de/publications/PerezDattari2019ICRA.pdf)

**Authors:** Rodrigo Pérez-Dattari, Carlos Celemin, Javier Ruiz-del-Solar, Jens Kober.

[Link to paper video](https://youtu.be/i4f1D4CH26E)


## Installation

To use this code it is necessary to insall python 3.6 and run it in a conda envrionment.
The code uses legacy tensorflow v1 code that is not compatible with python >= 3.7.

Since the code uses the MuJoCo Fetch environment, it is necessary to have MuJoCo installed to run this locally.


### Requirements
* setuptools==38.5.1
* numpy==1.13.3
* opencv_python==3.4.0.12
* matplotlib==2.2.2
* tensorflow==1.4.0
* pyglet==1.3.2
* gym== 0.15.4

## Usage

1. To run the main program type in the terminal (inside the folder `D-COACH`):

For running the Fetch Pick and Place Actor
```bash 
python main.py --use-pf --config-file 'pick_and_fetch'
```
For running the Fetch Slide Actor
```bash 
python main.py --use-fs --config-file 'fetch_and_slide'
```
For running the Fetch Push Actor
```bash 
python main.py --use-fp --config-file 'fetch_push'
```

Additionally, to train and save multiple models for the same domain append parser agrument ```-exp-num <index>``` when running main.py index is an int.

Trained models are stored in the graph directory, and training results like rewards and time per episode in the results directory. 

The default configuration files are unde the config_file directory and in the HD folder under the same directory Enhanced configuration files.

The trained models are stored under the graphs directory, to run a trained model: 
 1. Locate the .ini config file for the domain under the config_file directory
 2. Update the 'load_model' config to True
 3. Update the 'train' config to False
 4. Update the 'render' config to True
