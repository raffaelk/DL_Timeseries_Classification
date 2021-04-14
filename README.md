# Time Series Classification with Deep Learning

The code in this repo was part of a group project for the Deep Learning class of Thomas Hofmann at ETH Zürich. The corresponding 5-page hand in can be read [here](report_group_project.pdf). In this repo I present code which has been written by me. Most of it has also been used for parts of the group project, but here I created a stand-alone project with a slightly different focus.

Parts of the code which are based on or copied from other people's work are marked as such in comments.

## Summary

The goal of this project is to use deep learning to classify time series (e.g. sensor signals) into predefined categories. This is achieved by using three different network architectures:
- a fully convolutional net (FCN)
- a ResNet
- transformer

The FCN and the ResNet architecture have already been successfully used for such tasks (e.g. by [Fawaz et. al.](https://github.com/hfawaz/dl-4-tsc)). The transformer architecture on the other hand is widely used in the area of natural language processing, but it's application to time series classification is very rare.

All three architectures allow to create visualizations, which highlight important features in the signals. For the FCN and the ResNet so called class activation maps (CAM) have been used (as also described by [Fawaz et. al.](https://github.com/hfawaz/dl-4-tsc)). With the transformer architecture the attention values have been visualized, as it is sometimes done in natural language processing.

## Preparation to use the Code

## Description of the Code
