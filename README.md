# Assamese-Motif-Generation
This repository contains the Python implementation of a diffusion-based
generative framework for synthesizing traditional Assamese handloom motifs.
The work focuses on unconditional image generation using a curated cultural
motif dataset and evaluates generation quality using standard generative
model metrics.


## Overview

Traditional handloom motif design is a time-consuming and skill-intensive
process, often carried out manually using graph paper. This project explores
the use of diffusion models to assist in motif synthesis by learning structural
and symmetry patterns directly from existing designs.

The repository provides:
- Training code for the proposed architecture, Loom-GenNet integrated with DDIM
- Evaluation scripts for FID, KID, Precision, and Recall

##Dataset

LoomGen is a curated image dataset of traditional Assamese handloom motifs collected from Sualkuchi, Assam, and preprocessed digitally with the help of Photoshop and digital pen/stylus to improve clarity of images and for precise training.

Dataset Details
-Total images: 1212
-Image size: 512px × 512px
-Format: JPG
-Labels: None
-Intended for generative modeling

**Dataset link:**
**https://huggingface.co/datasets/bitupandeka/LoomGen**  

Please refer to the dataset card on Hugging Face for detailed metadata,
license, and usage terms.
