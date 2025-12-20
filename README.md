Monocular Depth Estimation for Screw Tightness State Detection

A low-cost, vision-based method for detecting industrial bolt tightness states using only a single RGB camera. It combines monocular depth estimation with a multi-feature fusion network for precise, marker-free inspection.
This warehouse serves as an explanation and resource for the supplementary experiments of this paper.

Core Method

Depth Estimation: Uses Depth Anything V2 to infer scene depth from one image.
Feature Extraction: Calculates the normalized depth difference between the bolt and its surroundings to create a robust feature.
Fusion & Regression: A custom SCR_R4Net network fuses RGB and depth features to predict the bolt-to-surface distance for tightness classification.

Supplementary Experiments Introduction

Comparative experiment on depth channel initialization strategies aims to verify the impact of initializing the newly added depth input channel with the weights of different RGB channels from the pre-trained ResNet-18 model on the training convergence speed and final regression accuracy of the model under the SCR_R4Net network architecture.

Table Performance Comparison of Depth Channel Initialization Strategies

Initialization Method	MSE	Convergence 	Epochs

R channel	       0.0083   	0.105	       37

G channel	       0.0085	    0.112	       82

B channel        0.0079   	0.121        77

Model robustness verification experiment aims to systematically evaluate the performance stability of the proposed screw tightness detection method under simulated complex real industrial environments. By simulating various common interferences such as illumination changes, oil contamination, and occlusion, the degree of performance degradation of the model is quantitatively analyzed, thereby clarifying the effectiveness and reliability boundaries of the method in practical deployment.
![image](https://github.com/qingshan2000/Depth-Estimation-for-Screw/blob/main/Supplementary%20experiment/Experimental%20data%20and%20supporting%20evidence/Interference%20dataset.png)

Qualitative analysis of failure cases and classification confusion research aims to gain an in-depth understanding of the failure modes and inherent causes of the model through qualitative analysis of its misprediction cases. At the same time, by constructing a confusion matrix for classification tasks, the main confusions of the model in distinguishing different tightness levels are systematically revealed, thereby clarifying the challenges faced by the current method in terms of perception limits.

![image](https://github.com/qingshan2000/Depth-Estimation-for-Screw/blob/main/Supplementary%20experiment/Experimental%20data%20and%20supporting%20evidence/Interference%20sample%20confusion%20matrix.png)


![image](https://github.com/qingshan2000/Depth-Estimation-for-Screw/blob/main/Supplementary%20experiment/Experimental%20data%20and%20supporting%20evidence/Invalid%20sample.png)

Depth estimation uncertainty and error propagation analysis experiment aims to quantitatively evaluate the errors introduced by the upstream pre-trained depth estimation model (Depth Anything V2) in this specific task, systematically analyze how these errors propagate through the subsequent processing chain, and finally clarify the error suppression mechanism that enables the entire system to achieve high-precision prediction without fine-tuning the depth model.
![image](https://github.com/qingshan2000/Depth-Estimation-for-Screw/blob/main/Supplementary%20experiment/Experimental%20data%20and%20supporting%20evidence/Visualization%20of%20depth%20estimation%20at%20different%20heights.png)
                                                       Visualization of depth estimation at different heights
Repository Contents

Interference Dataset.zip is a dataset for robustness experiments.
The supplementary experiment folder contains relevant experiment reports and verification materials.
For more information, please contact qs2026131691@163.com.
