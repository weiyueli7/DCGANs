# DCGANs

## Description

Our project aims to revolutionize image synthesis and classification by combining Deep Convolutional Generative Adversarial Networks (DCGANs) with Convolutional Neural Networks (CNNs). We seek to demonstrate the potential of DCGAN techniques in producing highly realistic images and achieving strong performance in image classification by utilizing these synthetic images during training. For this purpose, we selected the [Intel Natural Scene](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset. After processing with the DCGANs model, we generated natural scene images that are extremely authentic. To assess the performance of our method, we fed the synthetic images into a pre-trained ResNet-18 model and conducted the classification task using actual testing data. Our approach yielded impressive results, achieving a 78.1% 6-class classification accuracy.

## Getting Started

### Files
* `data.py`: creates dataloader
* `fake_image.py`: generates fake images
* `main.py`: performs image classification task
* `model.py`: implements the DCGANs model
* `resnet.py`: implements the ResNet model
* `zip.py`: zips the images

### Data
The dataset and associated structures for this project can be downloaded from [here](https://drive.google.com/file/d/1alGshdcbqKEi5mD6DgQmPHgCNCMPUae8/view?usp=sharing) and [here](https://drive.google.com/file/d/1L5EchJ2qhURek0Q1ub4gFQhyEyjTOHBC/view?usp=sharing).


### Executing program

To generate fake images, run:


```ruby
python3 fake_image.py
```

To perform the classification task, run:

```ruby
python3 main.py
```

Both files can take arguments. For more information, see the ArgumentParser within the files.

## Authors

Contributors (alphabetical order):

* Li, Weiyue - wel019@ucsd.edu
* Li, Yi - yil115@ucsd.edu

