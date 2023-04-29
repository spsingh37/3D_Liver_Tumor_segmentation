# 3D_Liver_Tumor_segmentation
In this project, we have compared Liver Tumor segmentation accuracies of four different architectures- UNet, ResUNet, SegResNet, & UNETR, over 2017 LiTS dataset. To evaluate the architectures' performances we used DICE score. 

# Dataset
The dataset is available for download on https://drive.google.com/drive/folders/13gtsM4-iFiBd_8cMKvIO7Q73d-YcdB0H?usp=share_link . Place this dataset in the "data" directory with the following structure.

data-><br>
----images-><br>
----------volume-0.nii<br>
----------....<br>
----------volume-130.nii<br>
----segmentations-><br>
----------segmentation-0.nii<br>
----------....<br>
----------segmentation-130.nii<br>
# MONAI & dependencies Installation
To install monai:<br>
pip install monai<br>

Then install some necessary dependencies:<br>
git clone https://github.com/Project-MONAI/MONAI.git <br>
cd MONAI/ <br>
pip install -e '.[nibabel,skimage]' <br>

# Training & Inference
To train the four architectures, run the "train_two_class.py" where the specific model to train can be passed as an argument. Also, the notebook "UNETR_LiTS_segmentation_3d.ipynb" can be only be used for training UNETR model, however, this notebook can be used to visualize the segmentation results for all the four achitectures.
![Screenshot](unet_loss_graph.png)
![Screenshot](resunet_loss_graph.png)
![Screenshot](segresnet_epoch.png)
![Screenshot](unetr_loss_plots.jpg)

# Results
![Screenshot](infer_small_tumor.png)
![Screenshot](infer_large_tumor.png)
