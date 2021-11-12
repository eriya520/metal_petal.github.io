# Flower Classification using TPUs

## Objective

Kaggle has a challenge to develop model for flower classification using TPUs. There are over 400,000 different types of flowers in the world, the challenge is limited to 104 different varieties of flowers.

The data are consisted of : 
* train: 12753 Tfrecord labelled train images
* val:   3712 Tfrecord labelled validation images
* test:  7382 Tfrecord unlabelled test images

Image size used in this project: 
* IMAGE_SIZE = (331, 331)

## Approach

In order to utilize TPUs, this project will use Google Colab to best utilize the Tensor Processing Units(TPUs) accelerator, data were prestored in buckets via Google Cloud Storage(GCS).

* Data EDA:
    * Define functions to load and unbatch images and labels in train and val datasets, and pre-process the data by normalize images, apply image augmentation (Random increase Saturation/Contrast, Random flip the images)
    * Optimize batch size based on TPU strategy.
    * Define function to display images from train and val datasets, verify the effect of image augmentation
* Modeling
    * Baseline CNN model with dropout layer but no fine tuning on optimizer
    * Use pretrained `Xception` models and `AverageGlobalPooling2D()` layer and explore the impact of various hyper parameters:
        * with/without drop out layer(s) and drop rate
        * with/without additional image augmentation (Random Rotation, Random flip)
        * three different learning rate algorithms (Time-based decay, Step-wise decay and self-defined LR)
* Make predictions on validation
    * compare overall performances of models
    * summarize the impact of various hyper parameters
* Model evaluation
    * Store the actual and predicted labels in a dataframe
    * Display images with predicted class and true class
    * Define a function to plot confusion matrix with a given amount of images in validation dataset

## Conclusion

* This Capstone demonstrated how to ultilize Tensor Processing Unit (TPU), a distribution strategy that TensowFlow specialized in deep learning tasks by powerful TPU cores.
* Xception pretrained model has significantly better performance than baseline CNN model(without fine-tuning).
  * Resources suggested that if we can play with the optmizer and learning rate, there is a chance we can improve the CNN model accuracy from 23%; however, it is interesting to learn that there are fundamental difference between Global Aaverage Pooling and fully connected layer as in CNN.[resources]('https://codelabs.developers.google.com/codelabs/keras-flowers-tpu#11')



* Fine-tuning parameters and augmentation are utilized in the deep learning using pre-trained Xception models with different learning rate scheduler and dropout layer.

|Model nomenclature|pretrain_model|Dropout_layers|Image_augmentation|LR algorithms|Overfit|Val_accuracy|
|---|---|---|---|---|---|---|
|conv2Ddrop|na|1|Y|adam default|extremely overfit|23%|
|time_model_aug|Xception|1|Y|Time-decay_LR|slightly overfit|85%|
|time_model|Xception|1|N|Time-decay LR|greatly overfit|84%|
|time_model_3|Xception|0|N|Time-decay LR|extremely overfit|83%|
|time_model_2|Xception|2|N|Time-decay LR|greatly overfit|83%|
|step_model|Xception|1|N|Step-wise decay LR|slightly overfit|85%|
|step_model_2|Xception|2|N|Step_wise decay LR|greatly overfit|85%|
|lrfn_model|Xception|1|N|Self-defined decay LR|slightly overfit|85%|
|lrfn_model_2|Xception|2|N|Self-defined decay LR|greatly overfit|85%|

### Summary of model performances
with self-defined learning rate schedulers were explored to construct deep learning model for multiclass classification. All LR algorithms have similar val accuracy (83-85%), but models with time_based decay and step_wise decay able to reach optimum val performance at less epochs than lrfn_decay.
* **dropout_layer**: 1 dropout layer with drate_rate = 0.5 is the best params
* **image_augmentation**: additional image_augmentation such as random rotation and random flip does not further impact on the validation accuracy.
* **learning rate**: time-based and step-wise decay both worked similar, lrfn_decay has lower val_performance at less epochs (epochs <10). 


## Credits

* Credits to the following resources which inspired and educated me 
    * Special Thanks to **Caroline S.**  for consulting and tips
    * GCS connection with Colab instructions [link](https://colab.research.google.com/notebooks/snippets/accessing_files.ipynb)
    * Tensorflow callbacks documentation [Documentation link](https://www.tensorflow.org/guide/keras/custom_callback)
    * Tensorflow image processing documentation [Documentation link](https://www.tensorflow.org/tutorials/images/data_augmentation)
    * Modeling using Keras image recognition pretrained model [Robert Border](https://www.kaggle.com/rborder/tpu-flower-classification?kernelSessionId=78320658)[, Umar Farooq](https://medium.com/@imUmarFarooq/computer-vision-petals-to-the-metal-3465d66ad343)
    * Learning rate scheduler and callback functions [Bachr Chi](https://medium.com/@bechr7/learning-rate-scheduling-with-callbacks-in-tensorflow-e2ba83647013) [Udacity PyTordh Chllengers](https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20)