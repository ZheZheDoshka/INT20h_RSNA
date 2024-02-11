### RSNA Pneumonia Detection Challenge

- model proposed in https://www.mdpi.com/2079-9292/10/13/1512 was used as a base for classification part of the task
- for object detection py-torches Faster R-CNN with MobileNetV3-Large backbone was used
- this is for VRAM economy
- no additional augmentation of images was done
- to replecate the results of training putting dataset in kaggle_set and creating and running the notebook CNN.ipynb should be enough
- to replicate the results of testing starting 'inference.py' with path to testing dataset folder should be enough
- python 3.10 was used
### Running inference.py
- Ensure that you have python 3.10
- Clone this repository git clone https://github.com/ZheZheDoshka/INT20h_RSNA
- Go to repository
- If you don't have virtualenv, install it pip install virtualenv
- Create new env virtualenv venv
- Enter it source venv/bin/activate
- Install requirement pip install -r requirements.txt
- From environment, run following: python inference.py --path path
where path - path to your dataset

#### Files in repository
- inference.py - used to get prediction boxes for test dataset
- model.py - contains functions for model creation and loading
- train.py - contains functions for model training
- dataset.py - contains functions for dataset and dataloader creation
- converter.py - contains functions for creation of dataframes for datasets and image conversions
- CNN.ipynb - main notebook where everything was done and everything is sure to work