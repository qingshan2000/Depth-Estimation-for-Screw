Monocular Depth Estimation for Screw Tightness State Detection

A low-cost, vision-based method for detecting industrial bolt tightness states using only a single RGB camera. It combines monocular depth estimation with a multi-feature fusion network for precise, marker-free inspection.
This warehouse serves as an explanation and resource for the supplementary experiments of this paper.

Core Method

Depth Estimation: Uses Depth Anything V2 to infer scene depth from one image.
Feature Extraction: Calculates the normalized depth difference between the bolt and its surroundings to create a robust feature.
Fusion & Regression: A custom SCR_R4Net network fuses RGB and depth features to predict the bolt-to-surface distance for tightness classification.

Supplementary Experiments 

![image](https://github.com/qingshan2000/Depth-Estimation-for-Screw/blob/main/Supplementary%20experiment/Experimental%20data%20and%20supporting%20evidence/Invalid%20sample.png)

Repository Contents

Interference Dataset.zip is a dataset for robustness experiments.
The supplementary experiment folder contains relevant experiment reports and verification materials.
For more information, please contact qs2026131691@163.com.
