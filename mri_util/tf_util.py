"""Common functions for setup."""
import datetime

import numpy as np
import pydicom
import scipy.signal
import tensorflow as tf
from pydicom.dataset import Dataset, FileDataset


def compute_psnr(predictions, ground_truths, maxpsnr=100):
    """Compute PSNR."""
    ndims = len(predictions.get_shape().as_list())
    mse = tf.reduce_mean(
        tf.square(tf.abs(predictions - ground_truths)), axis=list(range(1, ndims))
    )
    maxvals = tf.reduce_max(tf.abs(ground_truths), axis=list(range(1, ndims)))
    psnrs = (
        20 * tf.log(maxvals / tf.sqrt(mse)) / tf.log(tf.constant(10, dtype=mse.dtype))
    )
    # Handle case where mse = 0.
    psnrs = tf.minimum(psnrs, maxpsnr)
    return psnrs


def complex_to_channels(image, name="complex2channels"):
    """Convert data from complex to channels."""
    with tf.name_scope(name):
        image_out = tf.stack([tf.real(image), tf.imag(image)], axis=-1)
        # tf.shape: returns tensor
        # image.shape: returns actual values
        shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1] * 2]], axis=0)
        image_out = tf.reshape(image_out, shape_out)
    return image_out


def channels_to_complex(image, name="channels2complex"):
    """Convert data from channels to complex."""
    with tf.name_scope(name):
        image_out = tf.reshape(image, [-1, 2])
        image_out = tf.complex(image_out[:, 0], image_out[:, 1])
        shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1] // 2]], axis=0)
        image_out = tf.reshape(image_out, shape_out)
    return image_out


def circular_pad(tf_input, pad, axis):
    """Perform circular padding."""
    shape_input = tf.shape(tf_input)
    shape_0 = tf.cast(tf.reduce_prod(shape_input[:axis]), dtype=tf.int32)
    shape_axis = shape_input[axis]
    tf_output = tf.reshape(tf_input, tf.stack((shape_0, shape_axis, -1)))

    tf_pre = tf_output[:, shape_axis - pad :, :]
    tf_post = tf_output[:, :pad, :]
    tf_output = tf.concat((tf_pre, tf_output, tf_post), axis=1)

    shape_out = tf.concat(
        (shape_input[:axis], [shape_axis + 2 * pad], shape_input[axis + 1 :]), axis=0
    )
    tf_output = tf.reshape(tf_output, shape_out)

    return tf_output


def fftshift(im, axis=0, name="fftshift"):
    """Perform fft shift.

    This function assumes that the axis to perform fftshift is divisible by 2.
    """
    with tf.name_scope(name):
        split0, split1 = tf.split(im, 2, axis=axis)
        output = tf.concat((split1, split0), axis=axis)

    return output


def ifftc(im, name="ifftc", do_orthonorm=True):
    """Centered iFFT on second to last dimension."""
    with tf.name_scope(name):
        im_out = im
        dims = im_out.get_shape().as_list()
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * dims[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        tpdims = list(range(len(dims)))
        tpdims[-1], tpdims[-2] = tpdims[-2], tpdims[-1]

        im_out = tf.transpose(im_out, tpdims)
        im_out = fftshift(im_out, axis=-1)

        with tf.device("/gpu:0"):
            im_out = tf.ifft(im_out) * fftscale

        im_out = fftshift(im_out, axis=-1)
        im_out = tf.transpose(im_out, tpdims)

    return im_out


def fftc(im, name="fftc", do_orthonorm=True):
    """Centered FFT on second to last dimension."""
    with tf.name_scope(name):
        im_out = im
        dims = im_out.get_shape().as_list()
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * dims[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        tpdims = list(range(len(dims)))
        tpdims[-1], tpdims[-2] = tpdims[-2], tpdims[-1]

        im_out = tf.transpose(im_out, tpdims)
        im_out = fftshift(im_out, axis=-1)

        with tf.device("/gpu:0"):
            im_out = tf.fft(im_out) / fftscale

        im_out = fftshift(im_out, axis=-1)
        im_out = tf.transpose(im_out, tpdims)

    return im_out


def ifft2c(im, name="ifft2c", do_orthonorm=True):
    """Centered inverse FFT2 on second and third dimensions."""
    with tf.name_scope(name):
        im_out = im
        dims = tf.shape(im_out)
        if do_orthonorm:
            fftscale = tf.sqrt(tf.cast(dims[1] * dims[2], dtype=tf.float32))
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        # permute FFT dimensions to be the last (faster!)
        tpdims = list(range(len(im_out.get_shape().as_list())))
        tpdims[-1], tpdims[1] = tpdims[1], tpdims[-1]
        tpdims[-2], tpdims[2] = tpdims[2], tpdims[-2]

        im_out = tf.transpose(im_out, tpdims)
        im_out = fftshift(im_out, axis=-1)
        im_out = fftshift(im_out, axis=-2)

        # with tf.device('/gpu:0'):
        im_out = tf.ifft2d(im_out) * fftscale

        im_out = fftshift(im_out, axis=-1)
        im_out = fftshift(im_out, axis=-2)
        im_out = tf.transpose(im_out, tpdims)

    return im_out


def fft2c(im, name="fft2c", do_orthonorm=True):
    """Centered FFT2 on second and third dimensions."""
    with tf.name_scope(name):
        im_out = im
        dims = tf.shape(im_out)
        if do_orthonorm:
            fftscale = tf.sqrt(tf.cast(dims[1] * dims[2], dtype=tf.float32))
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        # permute FFT dimensions to be the last (faster!)
        tpdims = list(range(len(im_out.get_shape().as_list())))
        tpdims[-1], tpdims[1] = tpdims[1], tpdims[-1]
        tpdims[-2], tpdims[2] = tpdims[2], tpdims[-2]

        im_out = tf.transpose(im_out, tpdims)
        im_out = fftshift(im_out, axis=-1)
        im_out = fftshift(im_out, axis=-2)

        # with tf.device('/gpu:0'):
        im_out = tf.fft2d(im_out) / fftscale

        im_out = fftshift(im_out, axis=-1)
        im_out = fftshift(im_out, axis=-2)
        im_out = tf.transpose(im_out, tpdims)

    return im_out


def sumofsq(image_in, keep_dims=False, axis=-1, name="sumofsq", type="mag"):
    """Compute square root of sum of squares."""
    with tf.variable_scope(name):
        if type == "mag":
            image_out = tf.square(tf.abs(image_in))
        else:
            image_out = tf.square(tf.angle(image_in))
        image_out = tf.reduce_sum(image_out, keep_dims=keep_dims, axis=axis)
        image_out = tf.sqrt(image_out)

    return image_out


def slwin(ks, win_size=3, name="slwin"):
    """Perform sliding window on k-space."""
    # input size: [batch, x, y, t, coils]
    # right now, default is to do sliding window along fourth dimension

    with tf.variable_scope(name):
        ksc = circular_pad(ks, pad=int(win_size / 2), axis=3)

        # first initialize while loop variables
        i0 = tf.constant(0)  # iterator
        avg0 = tf.slice(
            ksc, [0, 0, 0, 0, 0], [-1, -1, -1, win_size, -1]
        )  # select window
        mask0 = kspace_mask(avg0, dtype=tf.complex64)
        nsamp0 = tf.reduce_sum(mask0, axis=3, keepdims=True)
        avg0 = tf.reduce_sum(avg0, axis=3, keepdims=True) / (nsamp0 + 1e-6)

        # while loop condition: i.e. while i < x.shape[1]
        def cond(i, avgs):
            return tf.less(i, tf.shape(ks)[3] - 1)

        # body of while loop
        def body(i, avgs):
            i = i + 1
            avg = tf.slice(
                ksc, [0, 0, 0, i, 0], [-1, -1, -1, win_size, -1]
            )  # select window
            mask = kspace_mask(avg, dtype=tf.complex64)
            nsamp = tf.reduce_sum(mask, axis=3, keepdims=True)
            avg = tf.reduce_sum(avg, axis=3, keepdims=True) / (nsamp + 1e-6)
            avgs = tf.concat([avgs, avg], axis=3)
            return [i, avgs]

        # execute while loop
        rets = tf.while_loop(
            cond,
            body,
            loop_vars=[i0, avg0],
            shape_invariants=[
                i0.get_shape(),
                tf.TensorShape([None, None, None, None, None]),
            ],
        )

    # while loop also returns the iterator (i), so throw it away!
    return rets[1]


def conj_kspace(image_in, name="kspace_conj"):
    """Conjugate k-space data."""
    with tf.variable_scope(name):
        image_out = tf.reverse(image_in, axis=[1])
        image_out = tf.reverse(image_out, axis=[2])
        mod = np.zeros((1, 1, 1, image_in.get_shape().as_list()[-1]))
        mod[:, :, :, 1::2] = -1
        mod = tf.constant(mod, dtype=tf.float32)
        image_out = tf.multiply(image_out, mod)

    return image_out


def replace_kspace(image_orig, image_cur, name="replace_kspace"):
    """Replace k-space with known values."""
    with tf.variable_scope(name):
        mask_x = kspace_mask(image_orig)
        image_out = tf.add(
            tf.multiply(mask_x, image_orig), tf.multiply((1 - mask_x), image_cur)
        )

    return image_out


def kspace_mask(image_orig, name="kspace_mask", dtype=None):
    """Find k-space mask."""
    with tf.variable_scope(name):
        mask_x = tf.not_equal(image_orig, 0)
        if dtype is not None:
            mask_x = tf.cast(mask_x, dtype=dtype)
    return mask_x


def kspace_threshhold(image_orig, threshhold=1e-8, name="kspace_threshhold"):
    """Find k-space mask based on threshhold.

    Anything less the specified threshhold is set to 0.
    Anything above the specified threshhold is set to 1.
    """
    with tf.variable_scope(name):
        mask_x = tf.greater(tf.abs(image_orig), threshhold)
        mask_x = tf.cast(mask_x, dtype=tf.float32)
    return mask_x


def kspace_location(image_size):
    """Construct matrix with k-space normalized location."""
    x = np.arange(image_size[0], dtype=np.float32) / image_size[0] - 0.5
    y = np.arange(image_size[1], dtype=np.float32) / image_size[1] - 0.5
    xg, yg = np.meshgrid(x, y)
    out = np.stack((xg.T, yg.T))
    return out


def tf_kspace_location(tf_shape_y, tf_shape_x):
    """Construct matrix with k-psace normalized location as tensor."""
    tf_y = tf.cast(tf.range(tf_shape_y), tf.float32)
    tf_y = tf_y / tf.cast(tf_shape_y, tf.float32) - 0.5
    tf_x = tf.cast(tf.range(tf_shape_x), tf.float32)
    tf_x = tf_x / tf.cast(tf_shape_x, tf.float32) - 0.5

    [tf_yg, tf_xg] = tf.meshgrid(tf_y, tf_x)
    tf_yg = tf.transpose(tf_yg, [1, 0])
    tf_xg = tf.transpose(tf_xg, [1, 0])
    out = tf.stack((tf_yg, tf_xg))
    return out


def create_window(out_shape, pad_shape=10):
    """Create 2D window mask."""
    g_std = pad_shape / 10
    window_z = np.ones(out_shape[0] - pad_shape)
    window_z = np.convolve(
        window_z, scipy.signal.gaussian(pad_shape + 1, g_std), mode="full"
    )

    window_z = np.expand_dims(window_z, axis=1)
    window_y = np.ones(out_shape[1] - pad_shape)
    window_y = np.convolve(
        window_y, scipy.signal.gaussian(pad_shape + 1, g_std), mode="full"
    )
    window_y = np.expand_dims(window_y, axis=0)

    window = np.expand_dims(window_z * window_y, axis=2)
    window = window / np.max(window)

    return window


def kspace_radius(image_size):
    """Construct matrix with k-space radius."""
    x = np.arange(image_size[0], dtype=np.float32) / image_size[0] - 0.5
    y = np.arange(image_size[1], dtype=np.float32) / image_size[1] - 0.5
    xg, yg = np.meshgrid(x, y)
    kr = np.sqrt(xg * xg + yg * yg)

    return kr.T


def sensemap_model(x, sensemap, name="sensemap_model", do_transpose=False):
    """Apply sensitivity maps."""
    with tf.variable_scope(name):
        if do_transpose:  # kspace [x,y,coils] -> image [x,y,emaps]
            x = tf.expand_dims(x, axis=-2)
            x = tf.multiply(tf.conj(sensemap), x)
            x = tf.reduce_sum(x, axis=-1)
        else:  # image [x,y,emaps] -> k-space [x,y,coils]
            x = tf.expand_dims(x, axis=-1)
            x = tf.multiply(x, sensemap)
            x = tf.reduce_sum(x, axis=-2)
    return x


def model_forward(x, sensemap, name="model_forward"):
    """Apply forward model.

    Image domain to k-space domain (y = A x).
    """
    with tf.variable_scope(name):
        if sensemap is not None:
            x = sensemap_model(x, sensemap, do_transpose=False)
        x = fft2c(x)
    return x


def model_transpose(x, sensemap, name="model_transpose"):
    """Apply transpose model.

    k-Space domain to image domain (x = A^T y).
    """
    with tf.variable_scope(name):
        x = ifft2c(x)
        if sensemap is not None:
            x = sensemap_model(x, sensemap, do_transpose=True)
    return x


def write_dicom(
    dicom_output,
    filename,
    seriesName,
    patientName,
    series_uid,
    yslice,
    phase,
    case=0,
    index=1,
):
    dicom_filename = pydicom.data.get_testdata_files("MR_small.dcm")[0]
    ds = pydicom.dcmread(dicom_filename)
    # normalize to uint16
    # horos is 16 bit
    # dicom images must be stored as int
    # normalize between 100 and 1000
    newMax = 1000
    newMin = 100
    oldMax = np.max(dicom_output)
    oldMin = np.min(dicom_output)
    dicom_output = (dicom_output - oldMin) * (newMax - newMin) / (
        oldMax - oldMin
    ) + newMin
    dicom_output = dicom_output.astype(np.uint16)

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = ds.BitsStored - 1
    ds.SamplesPerPixel = 1
    ds.SmallestImagePixelValue = dicom_output.min()
    ds.LargestImagePixelValue = dicom_output.max()
    ds.PixelRepresentation = 1  # 0
    ds.PatientName = patientName
    ds.PixelData = dicom_output.tobytes()
    ds.PatientID = series_uid

    ds.Rows, ds.Columns = dicom_output.shape

    # ds.SeriesDescription = case
    ds.SeriesNumber = 1
    ds.SeriesInstanceUID = series_uid

    num_phases = 18
    ds.InstanceNumber = phase + (yslice - 1) * num_phases

    ds.save_as(filename)

    return ds


def write_dicom_new(
    dicom_output,
    filename,
    seriesName,
    patientName,
    series_uid,
    yslice,
    phase,
    case=0,
    index=1,
):
    dicom_filename = pydicom.data.get_testdata_files("MR_small.dcm")[0]
    # dicom_filename = pydicom.data.get_testdata_files()[0]
    # dicom_filename = "/home/ekcole/from_david/dataset_000/Sec_009.mag"
    ds = pydicom.dcmread(dicom_filename)

    # file_meta = Dataset()
    # file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    # file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    # file_meta.ImplementationClassUID = "1.2.3.4"

    # ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    # ds.is_implicit_VR = True
    # ds.PatientName = "Test^Firstname"
    # ds.PatientID = "123456"

    # normalize to uint16
    # horos is 16 bit
    # dicom images must be stored as int
    # normalize between 100 and 1000
    newMax = 1000
    newMin = 100
    oldMax = np.max(dicom_output)
    oldMin = np.min(dicom_output)
    dicom_output = (dicom_output - oldMin) * (newMax - newMin) / (
        oldMax - oldMin
    ) + newMin
    
    # dicom_output = np.abs(dicom_output)
    dicom_output = dicom_output.astype(np.uint16)

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = ds.BitsStored - 1
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 0

    ds.SmallestImagePixelValue = dicom_output.min()
    ds.LargestImagePixelValue = dicom_output.max()

    ds.StudyDate = patientName
    ds.SeriesDate = patientName
    ds.AcquisitionDate = patientName
    ds.ContentDate = patientName
    ds.StudyTime = patientName
    ds.SeriesTime = patientName
    ds.AcquisitionTime = patientName
    ds.ContentTime = patientName
    ds.InstitutionName = patientName
    ds.StationName = patientName
    ds.StudyDescription = patientName
    ds.SeriesDescription = patientName
    ds.ManufacturerModelName = patientName
    ds.ProductId = patientName
    ds.PatientName = patientName
    ds.PatientID = patientName
    ds.ProtocolName = patientName

    ds.PixelData = dicom_output.tobytes()
    ds.Rows, ds.Columns = dicom_output.shape

    # # ds.SeriesNumber = 1
    ds.SeriesInstanceUID = series_uid

    # slices
    ds.StackID = "0"
    ds.InStackPositionNumber = yslice
    ds.InstanceNumber = yslice

    # ds.ImagesInSeries = 5
    # ds.AccessionNumber

    # phases
    # num_phases = 18
    # ds.InstanceNumber = phase + (yslice - 1) * num_phases

    ds.save_as(filename)

    return ds
