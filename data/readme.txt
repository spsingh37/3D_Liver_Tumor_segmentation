File structure writeup

-data/
    # data preperation folders used in data_preparation.ipynb
    |-dicom_file/ # folder to hold dicomm series files transcribed from 3D Slicer software
        |-images/ # first folder holds dicomm serires from original CT scan that will be input to network
            |-liver_XX0/ # each dicomm series for each patient will be in a sperate folder (each folder needs to be manually created)
                |-IMG0000.dcm
                |-IMG0001.dcm
                ...
            |-liver_XX1/
                |-IMGXXX0.dcm
                |-IMGXXX1.dcm
                ...
            ...
        |-labels/ # second folder holds dicomm serires from segmentation mask that will be output from network
            |-liver_XX0/ # each dicomm series for each patient will be in a sperate folder (each folder needs to be manually created)
                |-IMG0000.dcm
                |-IMG0001.dcm
                ...
            |-liver_XX1/
                |-IMGXXX0.dcm
                |-IMGXXX1.dcm
                ...
            ...
    |-dicom_groups/ # folder to hold subdivided dicomm series files, moved from dicom_file
        |-images/ # first folder holds dicomm serires from original CT scan that will be input to network (empty at first, folders automatically generated)
            |-liver_XX0_0/ # subdivided dicomm series will be in a seperate folder
                |-IMG0000.dcm
                |-IMG0001.dcm
                ...
            |-liver_XX0_1/
                |-IMG00X0.dcm
                |-IMG00X1.dcm
                ...
            ...
        |-labels/ # second folder holds dicomm serires from segmentation mask that will be output from network (empty at first, folders automatically generated)
            |-liver_XX0_0/ # subdivided dicomm series will be in a seperate folder
                |-IMG0000.dcm
                |-IMG0001.dcm
                ...
            |-liver_XX0_1/
                |-IMG00X0.dcm
                |-IMG00X1.dcm
                ...
            ...
    |-nifti_files # folder to hold the reconverted dicomm series subdivided dicom_groups
        |-images # first folder holds nifti from original CT scan that will be input to network (will be empty when data is moved to task_data)
            |-liver_X0_Y0.nii.gz
            |-liver_X0_Y1.nii.gz
            |-liver_X0_Y2.nii.gz
            |-liver_X0_Y3.nii.gz
            ...
        |-labels # second folder holds nifti from segmentation mask that will be output from network (will be empty when data is moved to task_data)
            |-liver_X0_Y0.nii.gz
            |-liver_X0_Y1.nii.gz
            |-liver_X0_Y2.nii.gz
            |-liver_X0_Y3.nii.gz
            ...
    |-task_data # before training, network will pull training and testing data sets from this folder
        |-TestSegmentation # stores nifti files that will form the testing data outputs
            |-liver_X0_Y0.nii.gz
            |-liver_X3_Y5.nii.gz
            ...
        |-TestVolumes # stores nifti files that will form the testing data inputs 
            |-liver_X0_Y0.nii.gz
            |-liver_X3_Y5.nii.gz
            ...
        |-TrainSegmentation # stores nifti files that will form the training data outputs
            |-liver_X0_Y1.nii.gz
            |-liver_X0_Y2.nii.gz
            |-liver_X0_Y3.nii.gz
            ...
        |-TrainVolumes # stores nifti files that will form the training data inputs 
            |-liver_X0_Y1.nii.gz
            |-liver_X0_Y2.nii.gz
            |-liver_X0_Y3.nii.gz
            ...
    |-task_results # after training, network will push all outputs to this folder, including best model, logs and stats
        |-best_metric_model.pth
        |-loss_test.npy
        |-loss_train.npy
        |-metric_test.npy
        |-metric_train.npy

    