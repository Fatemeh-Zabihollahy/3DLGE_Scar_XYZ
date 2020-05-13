

These 8 scripts are written to segment left ventricle (LV) myocardial scar from 3D LGE CMRI. First, the myocardium must be delineated from
the images. To this end, LGE_Myo_XY, LGE_Myo_XZ, and LGE_Myo_YZ are used to train three U-Nets using 2D slices extracted from three orthogonal
directions including transversal, sagittal, and coronal. Then the 3DLGE_Myo_XYZ is employed to combine the trained models for LV myocardial
segmentation. The binary segmentation maps created for LV myocardium must be saved to be used in the next step for scar segmentation. 
Similarly, three networks are trained by implementing LGE_Scar_XY, LGE_Scar_XZ, and LGE_Scar_YZ to identify the boundaries of scar 
in LV myocardium. 3DLGE_Scar_XYZ combines the prediction results to generate the segmentation map of the LV scar.
