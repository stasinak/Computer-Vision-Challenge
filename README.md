# CERTH Computer-Vision-Challenge


This is a small CV project that needs to be implemented in Python (either Tensorflow or Pytorch). Below you will find a detailed description of the tasks that need to be completed. Feel free to ask any questions that might arise.

## Task Description

You need to perform the task of multiclass image classification in this [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. The dataset is located in the dataset folder. It is already preprocessed and ready to be used. We provide you with a (simple) architecture to implement. We are mostly interested in the methodologies and analysis rather than the results. The problem itself is quite easy and well studied. You have the freedom to decide how you will approach it as a whole (from the Exploratory Data Analysis (EDA) to the final analysis of the results). As part of the assessment is to choose the appropriate methods for data exploration, hyperparameters and metrics.


## Implementation Instructions

 The dataset is on hdf5 format and it contains 4 arrays: x_train, x_test (images) and y_train, y_test (labels). You can read the dataset and each array using the following code
```
file_train = h5py.File(FILE_PATH, 'r')
x_train = file_train['x_train'][()]
y_train = file_train['y_train'][()]
x_test = file_train['x_test'][()]
y_test = file_train['y_test'][()]
file_train.close()
```

* **EDA**
  * Provide some basic statistics and plots for the dataset (for instance plot some training examples) and a short analysis of the findings
  * We have already normalized the data. Is there any effect of not normalizing it?

* **Train the network**
  * Implement the following architecture from scratch (You can use either Pytorch or tensorflow)
      * A **single convolutional layer** with a small filter size (3,3) and 32 as number of filters followed by a (2,2) **max pooling layer**. The filter maps are then **flattened** to provide features to the classifier.
      * Between the feature extractor and the output layer, add a **dense layer** to interpret the features with 128 nodes.
      * Given that the problem is a multi-class classification task, an output **dense layer** with 10 nodes is required along with a softmax activation function. 
      * All layers will use the ReLU activation function.
      * Use a stochastic gradient descent optimizer with a learning rate of 0.01 and a momentum of 0.9. The categorical cross-entropy loss function will be optimized, suitable for multi-class classification and monitor the classification accuracy metric
  * **If training is not taking too much time**, try to optimize your network by finetuning other hyperparameters such as batch size, learning rate, different optimizer etc. and provide the a short analysis of how you choose the best model

  * **Otherwise, if training is too time consuming for more experiments**, train the network for default parameters with batch size 128

    * What could be a difference if you train for a smaller batch of size (for example 16) instead of 128?  
    * How would you choose the best model?
    * What is the effect of the learning rate 

  * Present a plot of the learning curve. How will you choose the “optimal” epoch for stopping the training? Consider the impact of training for more or less epochs. What do you think that would be the case?

* **Evaluation**
  * Choose the appropriate metrics for evaluation, provide a short explanation of why you chose each metric and an analysis on the results for the best model (qualitative and quantitative)
  * Implement a simple model as a baseline (for example random classifier) and compare the results with your best network. What is the point of using such baselines?


## Theoretical questions

* How could you change the architecture of the network to provide better results and/or face issues that may arise (overfitting, underfitting etc.) ?
  * **Optional**: If time permits try some of them
* Given that the dataset is quite small, what approach could you use to increase the dataset and provide a more diverse dataset for improving the results? 
  * **Optional**: If time permits, implement the approach
* What other methods can you think of in order to improve the results (not in terms of architecture). 
  * **Optional**: If time permits try some of them
* How would you change the network in order to be used for other tasks (for example semantic segmentation)
* Assuming that you only have two classes (e.g dogs and cats) in the dataset and the classes in the dataset are not equally distributed, but 90/10.
  * How will you describe the dataset in a word?
  * Is accuracy suitable for this kind of problem?
  * Shortly describe how you will approach this kind of dataset

## General Comments

* The results are not that important, we are mostly interested in the way you approach the problem, the metrics you will choose and the analysis
* Feel free to include any other question that would be interesting to answer during the experiments
* Please provide comments, not only during the analysis but also to the code
* If you do not have time to implement a subtask, give a theoretical description of how you will approach the issue
* The network is very shallow, therefore it will run fast on a CPU. In case you want to run it on a GPU, you can check [google Colab](https://colab.research.google.com/)
* In case you need more time, feel free to contact us.


## Submission Instructions
1. Fork this repository
2. Implement your solution in an editor of your choice. You should use Jupyter Notebook to present the experiments/results, but you can also have code in different files (for example helper functions as a module)
3. Push your solution to your repository
4. Send us a link to your repository


Good luck!!! 
