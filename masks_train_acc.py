from mri_util import mri_prep
acc_y = 2
acc_z = 7
total_acc = acc_y*acc_z
dir_out = "/home_local/ekcole/knee_masks_%d" % total_acc
print(dir_out)
mri_prep.create_masks(
    dir_out,
    shape_y=320,
    shape_z=256,
    verbose=True,
    acc_y=[acc_y],
    acc_z=[acc_z],
    shape_calib=1,
    variable_density=True,
    num_repeat=3,
)

acc_y = 7
acc_z = 2
total_acc = acc_y*acc_z
dir_out = "/home_local/ekcole/knee_masks_%d" % total_acc
print(dir_out)
mri_prep.create_masks(
    dir_out,
    shape_y=320,
    shape_z=256,
    verbose=True,
    acc_y=[acc_y],
    acc_z=[acc_z],
    shape_calib=1,
    variable_density=True,
    num_repeat=3,
)