#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
nnUNet_train \
3d_fullres \
nnUNetPlusPlusTrainerV2 \
Task602_Z2 \
0 \
--npz
