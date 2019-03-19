How to use our code :


- Download the datasets city_scapes and foggy_cityscapes with the following links.
- Create folders "/small_cityscapes" and "/small_cityscapes_foggy". Use img_reshaping.py to resized the datasets. Change the folder and folder_reshaped variables with the path of the original dataset and the new path to load the reshaped images.


Before running the foggy.ipynb, you should have in your root :
		- A folder "/import" with datasets.py and fog_datasets.py. 
		- A folder "/models" with aspp.py, deeplab_plus.py, deeplabv3.py, functions.py, resnet.py and utils.py. 
		- A folder "/pretrained_models" with model_13_2_2_2__epoch_580.pth, resnet18-5c106cde.pth, resnet34-333f7ec4.pth, resnet50-19c8e357.pth.   
		- A folder "/small_cityscapes" with the resized images separated in two sets "val" and "train".
		- A folder "/small_cityscapes_foggy" with the same resized images with the synthetic fog.
		- A folder "/small_meta" with the labelled images.

Then run all cells of the notebook foggy.ipynb.

