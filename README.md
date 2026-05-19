# Microbubble-Velocity-Tracker

Given an ultrasound video, this code will attempt to detect bubbles, measure their velocities, and plot a distribution of the velocities

## Setup

First clone this repository onto your device. In order to do this, your device will need to have github installed on it. Navigate to a directory of your choosing and run the command `git clone [insert link here]`. This will copy all of the code into your chosen folder.

Before running this program, you will first want to ensure that python and all of the required packages are installed on your machine. I would recommend starting a virtual environment before installing the required packages, which can be done using the command `python -m venv .venv` and activated with `.venv\Scripts\activate`. After starting the virtual environment, you can install the required packages using `pip install -r requirements.txt`. Once all of the requirements are installed, you can run the code itself using `python main.py`.

## Running the Program

Once running the program, the code will prompt the user to select a video file. Once the file is selected, you will be asked to input a scale of your choice. A new window will open where you should drag to create a box or region of interest where the speeds will be measured. Once you select one, press enter and the code will run. Upon completion, a plot of the velocity distributions will be generated in the code folder.
