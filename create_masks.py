from mri_util import mri_prep

dir_out = "/home_local/ekcole/knee_masks"
mri_prep.create_masks(
    dir_out,
    shape_y=320,
    shape_z=256,
    verbose=True,
    acc_y=(1, 2, 3, 4),
    acc_z=(1, 2, 3, 4),
    shape_calib=1,
    variable_density=True,
    num_repeat=4,
)
