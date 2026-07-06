# Population-Scale Segmentation of Penile Tissue in DIXON MRI using Deep Learning

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Deep Learning Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## Overview

This repository contains the implementation of the research paper **"Population-Scale Segmentation of Penile Tissue in DIXON MRI using Deep Learning for Quantitative Phenotyping in Male Reproductive Health"** by Jan Ernsting et al. The paper introduces a novel deep learning framework for automated segmentation of penile tissue in multi-channel DIXON MRI scans. The approach leverages a 3D nnU-Net architecture optimized using a curated dataset of expert-annotated MRI slices.

### Core Idea
Quantitative assessment of penile anatomy is clinically significant for male reproductive and urogenital health. Traditional methods rely on external measurements, which are prone to variability and fail to capture internal structures. This study addresses these limitations by introducing a deep learning-based segmentation model that:
- Achieves observer-level accuracy in segmenting penile tissue.
- Enables high-throughput phenotyping at a population scale.
- Demonstrates high inter-session reproducibility in longitudinal studies.

The trained model was deployed on over **34,000** UK Biobank participants, providing a scalable solution for MRI-based penile anatomy assessment.

---

## How It Works

The implementation follows these key steps:

### 1. **Dataset Preparation**
   - A curated dataset of **13,050 expert-annotated slices** from **145 subjects** was used for training.
   - An independent test benchmark with **2,160 double-annotated slices** from **24 subjects** was employed for validation.

### 2. **Model Architecture**
   - The framework is built upon the **3D nnU-Net** architecture, which is tailored for medical image segmentation tasks.
   - The model was optimized for multi-channel DIXON MRI data, leveraging intensity differences between water and fat channels.

### 3. **Training and Validation**
   - The model achieved a **Dice score of 0.90** during 5-fold cross-validation and **0.92** on the independent test set.
   - Hausdorff distance was minimized to **3.58**, indicating precise boundary delineation.

### 4. **Deployment**
   - The trained model was applied to **34,412 UK Biobank participants** for automated segmentation and volumetric quantification.
   - Longitudinal evaluation in **2,282 men** demonstrated high reproducibility (**r = 0.87**).

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch (>= 1.9.0)
- NumPy
- nibabel (for handling MRI data)
- scikit-image
- tqdm

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/username/population-scale-penile-segmentation.git
cd population-scale-penile-segmentation
pip install -r requirements.txt
```

---

## Usage

### Running the Segmentation Script
The main Python script, `implementation.py`, performs penile tissue segmentation on DIXON MRI scans. Follow the steps below to use it.

#### Input Requirements
- **MRI Data**: Provide multi-channel DIXON MRI scans in NIfTI format (`.nii`).
- **Model Weights**: Pre-trained nnU-Net weights are required. Download them from [model_weights.pth](https://link-to-model-weights).

#### Command-Line Usage
Run the script with the following command:
```bash
python implementation.py --input_dir /path/to/mri/scans --output_dir /path/to/save/results --model_weights /path/to/model_weights.pth
```

#### Arguments
- `--input_dir`: Directory containing the input MRI scans.
- `--output_dir`: Directory where the segmented results will be saved.
- `--model_weights`: Path to the pre-trained model weights.

#### Example
```bash
python implementation.py --input_dir ./data/mri_scans --output_dir ./results/segmented --model_weights ./weights/model_weights.pth
```

### Output
- Segmented MRI volumes saved in NIfTI format (`.nii`).
- Summary statistics (e.g., Dice score, volumetric measurements) saved as `.csv`.

---

## Results

### Performance Metrics
- **Dice Score**: 0.92 (observer-level accuracy)
- **Hausdorff Distance**: 3.58
- **Reproducibility**: r = 0.87 (longitudinal evaluation)

### Visualizations
Sample segmentation results are shown below:

| Input MRI Slice | Ground Truth | Predicted Segmentation |
|------------------|--------------|-------------------------|
| ![Input](https://via.placeholder.com/150) | ![Ground Truth](https://via.placeholder.com/150) | ![Prediction](https://via.placeholder.com/150) |

---

## Citation

If you use this repository or the trained model in your research, please cite:

```bibtex
@article{Ernsting2023PenileSegmentation,
  title={Population-Scale Segmentation of Penile Tissue in DIXON MRI using Deep Learning for Quantitative Phenotyping in Male Reproductive Health},
  author={Jan Ernsting, Gunnar Paul Kordes, Nils Johannaber, Lynn Ogoniak, Wolfgang Roll, Tim Hahn, Alexander Siegfried Busch, Benjamin Risse},
  journal={arXiv preprint arXiv:2607.02127v1},
  year={2023}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We thank the authors of the paper for their contributions to the field of male reproductive health and for providing the curated dataset and trained model weights.

---

## Contact

For questions or collaborations, please contact [email@example.com](mailto:email@example.com).