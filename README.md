<div align="center">
  <a href="https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation">
    <img src="cover.png" alt="Logo" width="" height="200">
  </a>

<h1 align="center">Medical Image Segmentation</h1>
</div>

## 1. Problem Statement
<div align="justify"> Medical Image Segmentation is a computer vision task that involves dividing a medical image into multiple segments. In this context, the task is the segmentation of healthy organs in medical scans, particularly in the gastrointestinal (GI) tract, to enhance cancer treatment. For patients eligible for radiation therapy, oncologists aim to deliver high doses of radiation using X-ray beams targeted at tumors while avoiding the stomach and intestines. The goal is to effectively segment the stomach and intestines in MRI scans to improve cancer treatment, eliminating the need for the time-consuming and labor-intensive process in which radiation oncologists must manually outline the position of the stomach and intestines, addressing the challenge of daily variations in the tumor and intestines' positions. </div>

<br/>
  <div align="center">
    <a href="https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation">
      <img src="./images/image.jpg" alt="" width="350" height="350" align="center">
    </a>
  </div>
<br/>

As shown in the figure, the tumor (pink thick line) is close to the stomach (red thick line). High doses of radiation are directed at the tumor while avoiding the stomach. The dose levels are represented by a range of outlines, with higher doses shown in red and lower doses in green.

<div align="justify"> The main challenge is to provide better assistance to patients. The issue lies in the tumor size, which often results in radiation X-rays inadvertently coming into contact with healthy organs. The segmentations must be as precise as possible to prevent any unintended harm to the patient. The problem is to develop a deep learning solution that automates the segmentation of the stomach and intestines in MRI scans of cancer patients who undergo 1-5 MRI scans on separate days during their radiation treatment. </div>

## 2. Related Works
| Date | Title                                                                                                               | Code                                                                                                                     | Link                                                         |
|------|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| 2015 | U-Net: Convolutional Networks for Biomedical Image Segmentation                                                     | [Code](https://github.com/milesial/Pytorch-UNet) [Code](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/) | [Link](https://arxiv.org/pdf/1505.04597v1.pdf)               |
| 2015 | SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation                                    |                                                                                                                          | [Link](https://arxiv.org/pdf/1511.00561v3.pdf)               |
| 2016 | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation                                | [Code](https://github.com/mattmacy/vnet.pytorch)                                                                         | [Link](https://arxiv.org/pdf/1606.04797v1.pdf)               |
| 2018 | UNet++: A Nested U-Net Architecture for Medical Image Segmentation                                                  | [Code](https://github.com/MrGiovanni/UNetPlusPlus)                                                                       | [Link](https://arxiv.org/pdf/1807.10165v1.pdf)               |
| 2019 | CE-Net: Context Encoder Network for 2D Medical Image Segmentation                                                   | [Code](https://github.com/Guzaiwang/CE-Net)                                                                              | [Link](https://arxiv.org/pdf/1903.02740.pdf)                 |
| 2022 | Medical Image Segmentation using LeViT-UNet++: A Case Study on GI Tract Data                                        | None                                                                                                                     | [Link](https://arxiv.org/pdf/2209.07515v1.pdf)               |
| 2023 | 3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers                                      | [Code](https://github.com/Beckschen/3D-TransUNet)                                                                        | [Link](https://arxiv.org/pdf/2310.07781.pdf)                 |
| 2023 | DA-TransUNet: Integrating Spatial and Channel Dual Attention  with Transformer U-Net for Medical Image Segmentation | [Code](https://github.com/sun-1024/da-transunet)                                                                         | [Link](https://arxiv.org/pdf/2310.12570v1.pdf)               |
| 2023 | GI Tract Image Segmentation with U-Net and Mask R-CNN                                                               | None                                                                                                                     | [Link](http://cs231n.stanford.edu/reports/2022/pdfs/164.pdf) |

## 3. The Proposed Method
<div align="justify"> U-Net is a popular and effective architecture for medical image segmentation tasks, including segmenting different parts of the gastrointestinal tract. It is known for its ability to produce accurate segmentations, especially when dealing with limited training data. The architecture of U-Net gives the model the ability of precise localization, meaning the model output a class label for each pixel, and therefore achieve image segmentation. 
<br/>
  <div align="center">
      <img src="./images/unet-architecture.png" alt="" width="500" height="" align="center">
    </a>
  </div>
<br/>

Figure shows network artitecture in the [original paper](https://arxiv.org/pdf/1505.04597v1.pdf). It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (RELU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.</div>

<br/>
  <div align="center">
      <img src="./images/train-block-diagram.png" alt="" width="650" height="" align="center">
    </a>
  </div>
<br/>

<div align="justify"> For this project, As shown in the figure, the model takes MRI scans from cancer patients as input images, then uses the U-Net method to obtain predicted segmented areas of patients' MRI scans for "stomach", "large bowel", and "small bowel". By employing the loss function, it compares the predicted mask to the true mask, which we aim to minimize.

The [evaluation](https://torchmetrics.readthedocs.io/en/stable/classification/dice.html) metric is used the Dice. </div>


## 4. Implementation
This section delves into the practical aspects of the project's implementation.

### 4.1. Dataset
Under this subsection, you'll find information about the dataset used for the medical image segmentation task. It includes details about the dataset source, size, composition, preprocessing, and loading applied to it.
[Dataset](https://drive.google.com/file/d/1-2ggesSU3agSBKpH-9siKyyCYfbo3Ixm/view?usp=sharing)

<div align="justify">The dataset is MRIs of patients  provided by the UW-Madison Carbone Cancer Center. Specifically, the dataset contains 85 cases with 38496 scan slices of organs represented in 16-bit grayscale PNG format. Each case is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Each scan slices is repeted 3 times, with large_bowel, small_bowel, and stomach classes. The annotations are provided in a csv format with the segmented areas represented as RLE-encoded masks. It would typically need to decode the RLE encoded masks to create pixel-wise binary masks.
An empty segmentation entry represents no mask presented for the class in the MRI scan slice. The dataset has missing values. 

The most common size across all images in the dataset is 266 × 266, and the rest are of sizes 310×360, 276×276, and 234×234 in frequency descending order. For this project, all images reshape to size 256 × 256. In order to feed the image, and mask into training models, process them as tensors. 

**Files**

*   **train.csv** - IDs and masks for all training objects.
*   **train.txt** - case IDs for training objects.
*   **validation.txt** - case IDs for validation objects.
*   **test.txt** - case IDs for test objects.
*   **train** - a folder of case/day folders, each containing slice images for a particular case on a given day.

**Columns of  train CSV file**

*   **id** - unique identifier for object
*   **class** - the predicted class for the object
*   **segmentation** - RLE-encoded pixels for the identified object

Split the dataset to training, validation, and testing sets based on provided text files. (data folder)

*  **train_data.csv**: IDs, class, segmentation, and image_paths for training objects.
*  **validation_data.csv**: IDs, class, segmentation, and image_paths for validation objects.
*   **test_data.csv**: IDs, class, segmentation, and image_paths for test objects.</div>

**Visualization of a batch of images and target masks in the training set**
<br/>
  <div align="center">
      <img src="./images/target_mask.gif" width="500" height="400" alt="" width="500" height="" align="center">
    </a>
  </div>
<br/>

### 4.2. Model
In this project, the [SegmentationModelsPytorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/) library is used along with Pytorch to ceate a UNet model.

**create segmentation model with pretrained encoder**<br/> 
in_channels = 1, # model input channels (1 for gray-scale images)<br/> 
classes = 3, # model output channels <br/> <br/> 
model = smp.Unet(encoder_name='efficientnet-b1',<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in_channels=1,<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;encoder_weights='imagenet',<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;classes=3,<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;activation=None)
                
### 4.3. Configurations
**Loss Function**</br>
> TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)</br>
> BCELoss = smp.losses.SoftBCEWithLogitsLoss()</br>
> loss_fn = 0.7 * TverskyLoss(y_pred, y_true) + 0.3 * BCELoss(y_pred, y_true)</br>

**Metric**</br>
> metric = torchmetrics.Dice(average='macro', num_classes=3).to(device)</br>

**Optimizer**</br>
> optimizer = optim.SGD(model.parameters(), lr=0.8, momentum=0.9, weight_decay=1e-4)

### 4.4. Train
Here, you'll find instructions and code related to the training of the segmentation model. This section covers the process of training the model on the provided dataset.

### 4.5. Evaluate
In the evaluation section, the methods and metrics used to assess the model's performance are detailed. It explains how the model's segmentation results are quantified and provides insights into the model's effectiveness.

