# Time Series Classification with Deep Learning

The code in this repo was part of a group project for the Deep Learning class of Thomas Hofmann at ETH Zürich. The corresponding 5-page hand in can be read [here](report_group_project.pdf). In this repo I present code which has been written by me. Most of it has also been used for parts of the group project, but here I created a stand-alone project with a slightly different focus.

Parts of the code which are based on or copied from other people's work are marked as such in comments.

## Main Technologies Used

- Language: Python
- Visualization: Matplotlib
- Analysis and Modelling: PyTorch, NumPy

## Summary

The goal of this project is to use deep learning to classify time series (e.g. sensor signals) into predefined categories, a task quite common in almost every business field. Here this is achieved by using three different neural network architectures:
- fully convolutional net (FCN)
- ResNet
- transformer

The FCN and the ResNet architecture have already been successfully used for such tasks (e.g. by [Fawaz et. al.](https://github.com/hfawaz/dl-4-tsc)). The transformer architecture on the other hand is widely used in the area of natural language processing, but it's application to time series classification is very rare.

All three architectures allow to create visualizations, which highlight important features in the signals. For the FCN and the ResNet so called class activation maps (CAM) have been used (as also described by [Fawaz et. al.](https://github.com/hfawaz/dl-4-tsc)). With the transformer architecture the attention values have been visualized, as it is sometimes done in natural language processing.

The notebook [walkthrough.ipynb](walkthrough.ipynb) (or [open with nbviewer](https://nbviewer.jupyter.org/github/raffaelk/DL_Timeseries_Classification/blob/main/walkthrough.ipynb)) will apply the three networks to a sample dataset.

## Dataset

The models used in this project should be applicable to a wide variety of time series. Therefore the publicly accessible UCR time series archive is a good test case.

- Download the URC data set from: https://www.cs.ucr.edu/~eamonn/time_series_data/
- Unzip the dataset in the code directory. The files should be in: /UCR_archive/UCR_TS_Archive_2015/"datasetname"


## Description of the Code
- The subdirectory [nets](nets) containes the network architectures.
- The subdirectory [utils](utils) contains functions to train the models and create visualizations.
- [main_train_save.py](main_train_save.py) trains a model and saves it for later use. Various parameters can be set in the script.
- [main_visualize.py](main_visualize.py) creates visualizations from a pretrained and saved model.
