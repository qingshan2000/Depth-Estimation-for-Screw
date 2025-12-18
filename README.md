Monocular Bolt Tightness Detection

A low-cost, vision-based method for detecting industrial bolt tightness states using only a single RGB camera. It combines monocular depth estimation with a multi-feature fusion network for precise, marker-free inspection.

Core Method

Depth Estimation: Uses Depth Anything V2 to infer scene depth from one image.
Feature Extraction: Calculates the normalized depth difference between the bolt and its surroundings to create a robust feature.
Fusion & Regression: A custom SCR_R4Net network fuses RGB and depth features to predict the bolt-to-surface distance for tightness classification.

Repository Contents

Interference Dataset.zip is a dataset for robustness experiments
