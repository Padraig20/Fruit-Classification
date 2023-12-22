# Fruit Classification

## Motivation

I was in Linz at Ars Electronica last week, and found a cool exhibition on Artificial Intelligence. There was one interactive station, which I really enjoyed. It featured a camera, feeding the input into a Convolutional Neural Network (CNN), trained for object detection. You could hold e.g. a figure of a tiger into the camera, and about 10 monitors showed how the image was passed through each layer of the network. Finally, the predictions along with the probabilities would show up on another monitor. And all that happens in real-time!

I found the idea incredibly cute and funny, so I wanted to do a simplified version of that myself.

![image](https://github.com/Padraig20/Fruit-Classification/assets/111874815/c70384a3-3abd-4236-a076-2b3870bc8bb1)

## Dataset and Architecture

I needed something I could easily test myself. So why not fruit? The fruits360 dataset (available publicly on [Kaggle][https://www.kaggle.com/datasets/moltean/fruits/data]) contains enough pictures to make a half decent model, which I can easily test at home!

For architecture, I need something that runs fast - after all, I do not have access to high-end GPUs myself. This is why I chose a Residual Neural Network (ResNet50) initialized on ImageNET. I added a few custom layers for normalization, so it would converge a little faster, with less computational power.

## Current Results

The current model is terrible. But hey, give it time, I only ran 2 epochs...
