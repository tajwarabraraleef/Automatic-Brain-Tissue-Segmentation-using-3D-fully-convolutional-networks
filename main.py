#Reference taken from https://github.com/joseabernal/iSeg2017-nic_vicorob
#Tajwar-Julia
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
import itertools
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Cropping3D
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.models import load_model

K.set_image_dim_ordering('th')


seed = 7
np.random.seed(seed)
#diefining parameters
num_classes = 4
patience = 1
model_filename = 'models/outrun_step_{}.h5'
csv_filename = 'log/outrun_step_{}.cvs'

nb_epoch = 50

# Architecture taken from Dolz, J. et al. 3D fully convolutional networks for subcortical segmentation in MRI :
# A large-scale study. Neuroimage, 2017.
def generate_model(num_classes) :
    init_input = Input((1, 27, 27, 27))

    x = Conv3D(25, kernel_size=(3, 3, 3))(init_input)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)

    y = Conv3D(50, kernel_size=(3, 3, 3))(x)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)

    z = Conv3D(75, kernel_size=(3, 3, 3))(y)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)

    x_crop = Cropping3D(cropping=((6, 6), (6, 6), (6, 6)))(x)
    y_crop = Cropping3D(cropping=((3, 3), (3, 3), (3, 3)))(y)

    concat = concatenate([x_crop, y_crop, z], axis=1)

    fc = Conv3D(400, kernel_size=(1, 1, 1))(concat)
    fc = PReLU()(fc)
    fc = Conv3D(200, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)
    fc = Conv3D(150, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)

    pred = Conv3D(num_classes, kernel_size=(1, 1, 1))(fc)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, 9 * 9 * 9))(pred)
    pred = Permute((2, 1))(pred)
    pred = Activation('softmax')(pred)

    model = Model(inputs=init_input, outputs=pred)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])
    return model




T1_vols = np.zeros((10, 310, 182, 310)) #Zero padding to compensate missing pixels
label_vols = np.zeros((10, 310,182, 310)) #doing same with GT

#reading the 10 training images
for case_idx in range(1, 11) :
    T1_vols[(case_idx - 1), 26:282, 26:154, 26:282] = read_vol(case_idx, '')
    label_vols[(case_idx - 1), 26:282, 26:154, 26:282] = read_vol(case_idx, '_seg')

#Preprocessing, intensity normalization
T1_mean = T1_vols.mean()
T1_std = T1_vols.std()
T1_vols = (T1_vols - T1_mean) / T1_std

#extracting patches from the training set
x_train, y_train = build_set(T1_vols, label_vols, (9, 9, 9)) #change depending on the amount of patches you want

#reading validation set of 5 images
xval = np.zeros((5, 310, 182, 310)) #zero padding
xlabel_vols = np.zeros((5, 310, 182, 310))
for case_idx in range(10, 15) :
    xval[(case_idx - 10),26:282, 26:154, 26:282] = read_vol(case_idx, '')
    xlabel_vols[(case_idx - 10), 26:282, 26:154, 26:282] = read_vol(case_idx, '_seg')
#Preprocessing, intensity normalization
xval_mean = T1_mean
xval_std = T1_std 
xval = (xval - xval_mean) / xval_std
#extracting patches from the training set
x_val, y_val = build_set(xval, xlabel_vols, (9, 9,9)) #change depending on the amount of patches you want


# to prevent over fitting
stopper = EarlyStopping(patience=patience)

# Model checkpoint to save the training results
checkpointer = ModelCheckpoint(
    filepath=model_filename.format(1),
    verbose=0,
    save_best_only=True,
    save_weights_only=True)

# CSVLogger to save the training results in a csv file
csv_logger = CSVLogger(csv_filename.format(1), separator=';')
callbacks = [checkpointer, csv_logger, stopper]


# Build model
model = generate_model(num_classes)

#Comment model.fit part if you only itend to do classification
model.fit(
     x_train,
     y_train,
     epochs=nb_epoch,
     batch_size = 32,
     validation_data=(x_val,y_val),
     verbose=2,
     callbacks=callbacks)

# freeing space
del x_train
del y_train



# Load the trained model

model = generate_model(num_classes)
model.load_weights(model_filename.format(1))

#Loading validation and testing set
for case_idx in range(11, 19):
    T1_test_vol = np.zeros((310, 182, 310))
    T1_test_vol[26:282, 26:154, 26:282]= read_vol(case_idx, '')[:256, :128, :256] #zero padding
    x_test = np.zeros((18432, 1, 27, 27, 27)) #162450 #18432 #8112
    x_test[:, 0, :, :, :] = extract_patches(T1_test_vol, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9)) #extracting patches
    x_test[:, 0, :, :, :] = (x_test[:, 0, :, :, :] - T1_mean) / T1_std
    pred = model.predict(x_test, verbose=2)
    pred_classes = np.argmax(pred, axis=2)
    pred_classes = pred_classes.reshape((len(pred_classes), 9, 9, 9))
    segmentation = reconstruct_volume(pred_classes, (310, 182, 310)) #reconstructing back to input size
    segmentation2 = np.zeros((256, 128, 256))
    segmentation2 = segmentation[26:282, 26:154, 26:282] #removing zero padding
    save_vol(segmentation2, case_idx) #saving volume

    print
    "Finished segmentation of case # {}".format(case_idx)

# General utils for reading and saving data
def get_filename(set_name, case_idx, input_name, loc='datasets'):
    pattern = '{0}/{1}/IBSR_0{2}{3}.nii.gz'
    return pattern.format(loc, set_name, case_idx, input_name)


def get_set_name(case_idx):
    return 'Training' if case_idx < 11 else 'Testing'


def read_data(case_idx, input_name, loc='datasets'):
    set_name = get_set_name(case_idx)

    image_path = get_filename(set_name, case_idx, input_name, loc)

    return nib.load(image_path)

def read_vol(case_idx, input_name, loc='datasets'):
    image_data = read_data(case_idx, input_name, loc)

    return image_data.get_data()[:, :, :, 0]

def save_vol(segmentation, case_idx, loc='results'):
    set_name = get_set_name(case_idx)
    input_image_data = read_data(case_idx, '')

    segmentation_vol = np.empty(input_image_data.shape)
    segmentation_vol[:256, :128, :256, 0] = segmentation

    filename = get_filename(set_name, case_idx, '_seg', loc)
    nib.save(nib.analyze.AnalyzeImage(
        segmentation_vol.astype('uint8'), input_image_data.affine), filename)

def extract_patches(volume, patch_shape, extraction_step):
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches,) + patch_shape)

def build_set(T1_vols, label_vols, extraction_step=(9, 9, 9)):
    patch_shape = (27, 27, 27) #27,27,27
    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, 1, 27, 27, 27))
    y = np.zeros((0, 9 * 9 * 9, num_classes))
    for idx in range(len(T1_vols)):
        y_length = len(y)

        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), 1, 27, 27, 27))))
        y = np.vstack((y, np.zeros((len(label_patches), 9 * 9 * 9, num_classes))))

        for i in range(len(label_patches)):
            tmp =np_utils.to_categorical(label_patches[i].flatten(), num_classes);
            y[i + y_length] = tmp 
        del label_patches

        # Sampling strategy: reject samples which labels are only zeros
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step)
        x[y_length:, 0, :, :, :] = T1_train[valid_idxs]
        del T1_train

        # Sampling strategy: reject samples which labels are only zeros

    return x, y

def generate_indexes(patch_shape, expected_shape):
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i + 1] * (expected_shape[i] // patch_shape[i + 1]) for i in range(ndims - 1)]

    idxs = [range(patch_shape[i + 1], poss_shape[i] - patch_shape[i + 1], patch_shape[i + 1]) for i in range(ndims - 1)]

    return itertools.product(*idxs)


def reconstruct_volume(patches, expected_shape):
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)):
        selection = [slice(coord[i], coord[i] + patch_shape[i + 1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img


