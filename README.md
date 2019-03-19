# MAP583SementicSegmentation

How to use our code :


- Download the datasets city_scapes and foggy_cityscapes with the following links.
For cityscapes, you need to create an account on their website, then you can download the databse we used here:  https://www.cityscapes-dataset.com/file-handling/?packageID=3
For foggy_cityscapes: https://data.vision.ee.ethz.ch/csakarid/shared/SFSU_synthetic/Downloads/Foggy_Cityscapes/leftImg8bit_trainvaltest_transmittance.zip
- Create folders "/small_cityscapes" and "/small_cityscapes_foggy". Use img_reshaping.py to resized the datasets. Change the folder and folder_reshaped variables with the path of the original dataset and the new path to load the reshaped images.


Before running the foggy.ipynb, you should have in your root :
		- A folder "/import" with datasets.py and fog_datasets.py. 
		- A folder "/models" with aspp.py, deeplab_plus.py, deeplabv3.py, functions.py, resnet.py and utils.py. 
		- A folder "/pretrained_models" with model_13_2_2_2__epoch_580.pth, resnet18-5c106cde.pth, resnet34-333f7ec4.pth, resnet50-19c8e357.pth. We could'nt upload it because it was too heavy, you can get it here: https://github.com/fregu856/deeplabv3/tree/master/pretrained_models  
		- A folder "/small_cityscapes" with the resized images separated in two sets "val" and "train".
		- A folder "/small_cityscapes_foggy" with the same resized images with the synthetic fog.
		- A folder "/small_meta" with the labelled images.

Then run all cells of the notebook foggy.ipynb.
