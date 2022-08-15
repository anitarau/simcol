# [Bimodal Camera Pose Prediction for Endoscopy](https://arxiv.org/abs/2204.04968)

[Access data](https://www.synapse.org/#!Synapse:syn28548633/wiki/617126)  

The data from our paper and sequences from additional anatomies can be accessed for the SimCol-to-3D MICCAI 2022 challenge.

<p align="center">
  <img src="assets/pointcloud_S_1.png" alt="Example data visualization" width="600" />
</p>

## ⚙ Getting started

Download the datasets from [Synapse](https://www.synapse.org/#!Synapse:syn28548633/wiki/617126). To visualize the data like above run `visualize_3D_data.py` with the updated path to the data. 

## 🌍 Submitting to the SimCol-to-3D MICCAI challenge

If you are a registered participant and would like to submit your results, please use the `docker_templates` and follow the official [submission guide](https://github.com/anitarau/simcol/blob/main/EndoVis-SimCol2022_submission_guide.pdf).

## 📊 Evaluation and challenge leader board

Tasks 1, 2, and 3 will be evaluated according to the scripts in `evaluation`. To evaluate your method on validation data, generate predictions according to the `docker_templates' and run
```
python evaluation/eval_synthetic_depth.py /path/to/predictions /path/to/groundtruth
```