#!/usr/bin/env python
# coding: utf-8
import dicom2nifti
import nibabel as nib

def dcm2nifti(src, dest, reorient_nifti=True):
	dicom2nifti.dicom_series_to_nifti(src, dest, reorient_nifti)
	return True
	
