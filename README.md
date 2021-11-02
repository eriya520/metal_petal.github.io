# Kaggle flower images classification using keras model

-------
Kaggle has a competition to identify over 104 classes of flowers. The data provided in the kaggle competition are Tfrecord datasets: labelled train images, lablled val images, unlablled test images datasets with different image sizes.
-------

## Project Objectives
The objective of this project is to:
1) Be able to read and load large Tfrecord datasets
2) Develop a high accuracy multiclassification model for image recognition over 104 classes of flowers
3) Summarize the learning after evaluating various models via different neural network algorithms

## Approach
1) Created a EDA.py for storing all the functions for image loading and augmentation
2) In the modeling notebook, focus on comparing three different models:
  * Conv2D+Maxpooling with 3 hidden layers
  * Conv2D+Maxpooling with dropout layers
  * GlobalAveragePooling2D with LearningRateScheduler and self-defined learning rate algorithms(Time-based decay, Exponential decay etc.)

3) Evaluate the model outcome and summarize the learning on the project

## Conclusion
To be updated

**Credits to following link for this notebook.**

Functions are based on the referenced notebook!

* [Kaggle metal to petal challenge](https://www.kaggle.com/c/tpu-getting-started)
* [Robert Border](https://www.kaggle.com/rborder/tpu-flower-classification?kernelSessionId=78320658)
* [Khoa Phung](https://www.kaggle.com/khoaphng/model-efficientnetb7?kernelSessionId=76625228)
* [Tensorflow load and proprocessing images](https://www.tensorflow.org/tutorials/load_data/images)
