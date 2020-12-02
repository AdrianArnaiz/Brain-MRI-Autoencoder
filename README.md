Brain-MRI-Autoencoder
=

[![made-with-python](https://img.shields.io/badge/Coded%20with-Python-21496b.svg?style=for-the-badge&logo=Python)](https://www.python.org/)
[![made-with-latex](https://img.shields.io/badge/Documented%20with-LaTeX-4c9843.svg?style=for-the-badge&logo=Latex)](https://www.latex-project.org/)
![GitHub repo size](https://img.shields.io/github/repo-size/AdrianArnaiz/Brain-MRI-Denoiser-Autoencoder?style=for-the-badge&logo=Github)
![Github code size](https://img.shields.io/github/languages/code-size/AdrianArnaiz/Brain-MRI-Denoiser-Autoencoder?style=for-the-badge&logo=Github)
![GitHub license](https://img.shields.io/github/license/AdrianArnaiz/Brain-MRI-Denoiser-Autoencoder?style=for-the-badge&logo=Github)
![Github Follow](https://img.shields.io/github/followers/AdrianArnaiz?style=social&label=Follow)

###### Deep Autoencoder for brain MRI

***********

**Master's Thesis. Master's in Data Science at Universitat Oberta de Catalunya.**

#### Author
* **Adrián Arnaiz Rodríguez** - [aarnaizr@uoc.edu](mailto:aarnaizr@uoc.edu)

#### Tutor: 
* **Dr. Baris Kanber** - [bkanber@uoc.edu](mailto:bkanber@uoc.edu)

***************
## Convolutional Autoencoder Architectures used
![](ArchitecturesDiagram.svg)

#### Experiments:
* **With Data Augmentation**:
    * ***2 experiments (MSE and DSSIM Loss) for each of the following architectures:***
        * Shallow residual autoencoder (full-pre)
        * Shallow residual autoencoder (full-pre) + L2 reg.
        * Skip connection autoencoder
        * Skip connection autoencoder + L2 reg.
        * Myronenko Autoencoder
        * **RESIDUAL-UNET** (proposed new improved architecture)

* Without Data Augmentation:
    * MSE Loss
        * Shallow residual autoencoder (original)
        * Shallow residual autoencoder (full-pre)
        * Shallow residual autoencoder (full-pre) + L2 reg.
        * Skip connection autoencoder
        * Myronenko Autoencoder
        * Myronenko Autoencoder + L2 reg.
