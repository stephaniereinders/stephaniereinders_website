+++
title = "Create training and test sets"

date = "2020-09-30"

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["Stephanie Reinders"]

# Feature publication
featured = true

# Featured image thumbnail (optional)
image_preview = ""

# Is this a selected publication? (true/false)
selected = true

# Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter the filename (excluding '.md') of your project file in `content/project/`.
#   E.g. `projects = ["deep-learning"]` references `content/project/deep-learning.md`.
projects = []

# Links (optional).
url_pdf = ""
url_preprint = ""
url_code = ""
url_dataset = ""
url_project = ""
url_slides = ""
url_video = ""
url_poster = ""
url_source = ""

# Custom links (optional).
#   Uncomment line below to enable. For multiple links, use the form `[{...}, {...}, {...}]`.
# url_custom = [{name = "Custom Link", url = "http://example.org"}]

# Does the content use math formatting?
math = true

# Does the content use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
image = ""
caption = "training and test sets"

+++

In this tutorial I show how to create training and test sets using <span style="font-family:Courier; font-size:12pt;">scikit-learn</span> in Python.


> * [Get Started](#get_started)
> * [Load the Iris Dataset](#load)
> * [Create Training and Test Sets](#create)
>     * [Choose the size of the training and test sets](#size)
>     * [Stratify by class](#stratify)
>     * [Set the random state](#random)
>     * [Other parameters](#other)
> * [References](#references)


---

## Get Started <a class="anchor" id="get_started"></a>
The first thing we do is import the modules that we will use. 


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
```

## Load the Iris Dataset <a class="anchor" id="load"></a>
The Iris dataset is one of the toy datasets included with <span style="font-family:Courier; font-size:12pt;">sklearn.datasets</span>. We load the Iris dataset.


```python
iris = load_iris()
```

To learn more about the Iris dataset we view the description. The full description is quite long, so we only display the first five hundred characters. (You can use the command <span style="font-family:Courier; font-size:12pt;">print(iris['DESCR'])</span> to view the full description.) 


```python
print(iris['DESCR'][:500])
```

    .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
                    
    


The description tells us that the dataset contains 3 classes and each class has 50 samples for a total of 150 samples. The dataset has four variables or features. We can see how the data is stored by viewing the dataset's keys.


```python
iris.keys()
```




    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])



The feature values are stored under <span style="font-family:Courier; font-size:12pt;">data</span>. The names of the classes are stored under <span style="font-family:Courier; font-size:12pt;">target_names</span>. The class of each sample is stored under <span style="font-family:Courier; font-size:12pt;">target</span>, where 0 represents class setosa, 1 represents class versicolor, and 2 represents class virginica.


```python
iris.target_names
```




    array(['setosa', 'versicolor', 'virginica'], dtype='<U10')




```python
iris.target
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])



## Create Training and Test Sets <a class="anchor" id="create"></a>
We use the function <span style="font-family:Courier; font-size:12pt;">train_test_split()</span> from <span style="font-family:Courier; font-size:12pt;">sklearn.model_selection</span> to psuedo-randomly assign samples to training and testing sets. More specifically, <span style="font-family:Courier; font-size:12pt;">train_test_split()</span> splits the feature values stored in <span style="font-family:Courier; font-size:12pt;">iris.data</span> into two sets: <span style="font-family:Courier; font-size:12pt;">data_train</span> and <span style="font-family:Courier; font-size:12pt;">data_test</span>. The class labels for the samples in <span style="font-family:Courier; font-size:12pt;">data_train</span> are in <span style="font-family:Courier; font-size:12pt;">target_train</span> and the class labels for <span style="font-family:Courier; font-size:12pt;">data_test</span> are in <span style="font-family:Courier; font-size:12pt;">target_test</span>.


```python
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target)
```

### Choose the size of the training and test sets <a class="anchor" id="size"></a>
By default, <span style="font-family:Courier; font-size:12pt;">train_test_split()</span> puts approximately 75\% of the samples in the training set and approximately 25\% in the test set.


```python
print('Number of samples of in the training set:',target_train.shape[0])
print('Number of samples of in the test set:',target_test.shape[0])
```

    Number of samples of in the training set: 112
    Number of samples of in the test set: 38


We can specify the size of the training set with the parameter <span style="font-family:Courier; font-size:12pt;">train_size</span>. For example, if we want to use 100 samples for training we set <span style="font-family:Courier; font-size:12pt;">train_size</span> equal to 100. The remaining 50 samples will be placed in the test set.


```python
data_train100, data_test100, target_train100, target_test100 = train_test_split(iris.data,
                                                                                iris.target,
                                                                                train_size=100)
print('Number of samples of in the training set:',target_train100.shape[0])
print('Number of samples of in the test set:',target_test100.shape[0])
```

    Number of samples of in the training set: 100
    Number of samples of in the test set: 50


Instead of telling the function the number of samples we want to use for training, we can also tell it the percentage of samples we would like to use. To do this, we set <span style="font-family:Courier; font-size:12pt;">train_size</span> equal to a number between 0.0 and 1.0. For example, if we want our training set to be comprised of 50\% of the samples, we set <span style="font-family:Courier; font-size:12pt;">train_size</span> equal to 0.5. The other 50\% of the samples will be placed in the test set.


```python
data_train50, data_test50, target_train50, target_test50 = train_test_split(iris.data, 
                                                                            iris.target,
                                                                            train_size=0.5)
print('Number of samples of in the training set:',target_train50.shape[0])
print('Number of samples of in the test set:',target_test50.shape[0])
```

    Number of samples of in the training set: 75
    Number of samples of in the test set: 75


### Stratify by class <a class="anchor" id="stratify"></a>
We can view the number of samples of each class that were placed in the training and test sets.


```python
train_class_counts = [(target_train==0).sum(), (target_train==1).sum(), (target_train==0).sum()]
test_class_counts = [(target_test==0).sum(), (target_test==1).sum(), (target_test==0).sum()]
print('Number of samples of each class in the training set:', train_class_counts)
print('Number of samples of each class in the test set:', test_class_counts)
```

    Number of samples of each class in the training set: [33, 39, 33]
    Number of samples of each class in the test set: [17, 11, 17]


We can also plot the number of samples from each class in the training and test sets.


```python
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].bar(iris.target_names, train_class_counts)
axs[0].set_title('Training Set')
axs[1].bar(iris.target_names, test_class_counts)
axs[1].set_title('Test Set')

for ax in axs.flat:
    ax.set(xlabel='class', ylabel='number of samples')

# Hide y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
```


![png](output_22_0.png)


The iris dataset has 50 samples from each class so each class represents 1/3 of the dataset. If we use the default settings, the function <span style="font-family:Courier; font-size:12pt;">train_test_split()</span> does not try to form a training set from the Iris dataset where each class represents 1/3 of the training set. If we want the training and test sets to have the same class proportations as the original dataset, we can use the <span style="font-family:Courier; font-size:12pt;">stratify</span> parameter. 


```python
data_train_strat, data_test_strat, target_train_strat, target_test_strat = train_test_split(iris.data,
                                                                                            iris.target,
                                                                                            stratify=iris.target)
```


```python
train_class_counts = [(target_train_strat==0).sum(), (target_train_strat==1).sum(),(target_train_strat==0).sum()]
test_class_counts = [(target_test_strat==0).sum(), (target_test_strat==1).sum(),(target_test_strat==0).sum()]
print('Number of samples of each class in the training set:', train_class_counts)
print('Number of samples of each class in the test set:', test_class_counts)
```

    Number of samples of each class in the training set: [37, 38, 37]
    Number of samples of each class in the test set: [13, 12, 13]


We can plot the number of samples from each class in our new training and test sets.


```python
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].bar(iris.target_names, train_class_counts)
axs[0].set_title('Stratified Training Set')
axs[1].bar(iris.target_names, test_class_counts)
axs[1].set_title('Stratified Test Set')

for ax in axs.flat:
    ax.set(xlabel='class', ylabel='number of samples')

# Hide y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
```


![png](output_27_0.png)


### Set the random state <a class="anchor" id="random"></a>
Often, we want our results to be reproducible. The function <span style="font-family:Courier; font-size:12pt;">train_test_split()</span> uses on a psuedo-random number generator to assign samples to the training and test sets. If we want the function to form the exact same training and test sets every time, we can use the <span style="font-family:Courier; font-size:12pt;">random_state</span> parameter by setting it equal to any non-negative integer that we choose. 


```python
data_train, data_test, target_train, target_test = train_test_split(iris.data,
                                                                    iris.target,
                                                                    random_state=5)
```

### Other parameters <a class="anchor" id="other"></a>
For a full list of parameters that can be used with <span style="font-family:Courier; font-size:12pt;">train_test_split()</span> go to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#.

---

## References <a class="anchor" id="references"></a>
*I used the following sources when writing this tutorial and found them to be quite helpful:*
> 1. https://scikit-learn.org/stable/datasets/index.html
> 2. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#
> 3. https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/categorical_variables.html
> 4. https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html
> 5. https://moonbooks.org/Articles/How-to-create-a-table-of-contents-in-a-jupyter-notebook-/
