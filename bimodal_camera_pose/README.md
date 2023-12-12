# [Bimodal Camera Pose Prediction for Endoscopy](https://arxiv.org/abs/2204.04968)

This repo contains the traiing and testing code. 

## ⚙ Getting started

To replicate the results in our Bimodal Camera Pose paper, download SyntheticColon_I [here](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763) (you do not need to download II and III). 

Download our pretrained model [here](https://drive.google.com/file/d/1aHWPqS1X8v-T2V9ssqO-qz7R-e6xf5GB/view?usp=share_link) and save it to `bimodal_camera_pose/trained_models/posenet_binned`. 


## Testing

```
python test.py /path/to/SyntheticColon_I --test-file /path/to/test_file.txt
```


## Training

TO DO!