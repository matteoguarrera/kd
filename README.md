# Proven Robust Knowledge Distillation

## Segmentation

Given the system property *distance from the center line*, a proxy property has been defined for simplicity. 
> Number of yellow pixel in the image.

Three models are under analysis:
 - The ground truth controller. Hardcoded module that counts the number of yellow pixel
 - A teacher that has been trained using the WeBots labels
 - Several students models that have been trained using the teacher as ground truth.

### Usage
1. Download the release files and put the model and the *.npy dataset in the segment folder.  
2. Play with the `UNET.ipynb` 


---
## Behavioral Cloning, still to improve

```bash

cd ~/Documents/kd/example1/controllers/follower/build && cmake .. && make
cd ~/Documents/kd/example1/controllers/lead/build && cmake .. && make
 
```