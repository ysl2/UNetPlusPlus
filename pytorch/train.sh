#!/bin/bash

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
nnUNet_train \
3d_fullres \
nnUNetPlusPlusTrainerV2 \
Task602_Z2 \
0 \
--npz
