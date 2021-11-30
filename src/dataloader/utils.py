# Import internal libraries
import src.config as config
import src.dataloader.augmentations as aug 
from src.dataloader.dataset import ZipDataset
from src.dataloader.han_miccai2015 import HaNMICCAI2015Dataset
from src.dataloader.han_deepmindtcia import HaNDeepMindTCIADataset

# Import external libraries
import os
import pdb
import itk
import copy
import time
import tqdm
import json
import urllib
import psutil
import pydicom
import humanize
import traceback
import numpy as np
import tensorflow as tf
from pathlib import Path
import SimpleITK as sitk # sitk.Version.ExtendedVersionString()

if len(tf.config.list_physical_devices('GPU')):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)



############################################################
#                    DOWNLOAD RELATED                      #
############################################################

class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_zip(url_zip, filepath_zip, filepath_output):
    import urllib
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=' - [Download]' + str(url_zip.split('/')[-1]) ) as pbar:
        urllib.request.urlretrieve(url_zip, filename=filepath_zip, reporthook=pbar.update_to)
    read_zip(filepath_zip, filepath_output)

def read_zip(filepath_zip, filepath_output=None, leave=False):
    
    import zipfile

    # Step 0 - Init
    if Path(filepath_zip).exists():
        if filepath_output is None:
            filepath_zip_parts     = list(Path(filepath_zip).parts)
            filepath_zip_name      = filepath_zip_parts[-1].split(config.EXT_ZIP)[0]
            filepath_zip_parts[-1] = filepath_zip_name
            filepath_output        = Path(*filepath_zip_parts)

        zip_fp = zipfile.ZipFile(filepath_zip, 'r')
        zip_fp_members = zip_fp.namelist()
        with tqdm.tqdm(total=len(zip_fp_members), desc=' - [Unzip] ' + str(filepath_zip.parts[-1]), leave=leave) as pbar_zip:
            for member in zip_fp_members:
                zip_fp.extract(member, filepath_output)
                pbar_zip.update(1)

        return filepath_output
    else:
        print (' - [ERROR][read_zip()] Path does not exist: ', filepath_zip)
        return None

def move_folder(path_src_folder, path_dest_folder):
    
    import shutil
    if Path(path_src_folder).exists():
        shutil.move(str(path_src_folder), str(path_dest_folder))
    else:
        print (' - [ERROR][utils.move_folder()] path_src_folder:{} does not exist'.format(path_src_folder))

############################################################
#                    ITK/SITK RELATED                      #
############################################################

def write_mha(filepath, data_array, spacing=[], origin=[]):
    # data_array = numpy array
    if len(spacing) and len(origin):
        img_sitk = array_to_sitk(data_array, spacing=spacing, origin=origin)
    elif len(spacing) and not len(origin):
        img_sitk = array_to_sitk(data_array, spacing=spacing)
        
    sitk.WriteImage(img_sitk, str(filepath), useCompression=True)

def write_mha(path_save, img_data, img_headers, img_dtype):

    # Step 0 - Path related
    path_save_parents = Path(path_save).parent.absolute()
    path_save_parents.mkdir(exist_ok=True, parents=True)

    # Step 1 - Create ITK volume
    orig_origin        = img_headers[config.KEYNAME_ORIGIN]
    orig_pixel_spacing = img_headers[config.KEYNAME_PIXEL_SPACING]
    img_sitk           = array_to_sitk(img_data, origin=orig_origin, spacing=orig_pixel_spacing)
    img_sitk           = sitk.Cast(img_sitk, img_dtype)
    sitk.WriteImage(img_sitk, str(path_save), useCompression=True)

    return img_sitk

def imwrite_sitk(img, filepath, dtype, compression=True):
    """
    img: itk (<class 'itk.itkImagePython.itkImageF3'>) or sitk image
    """
    import itk

    def convert_itk_to_sitk(image_itk, dtype):
        
        img_array = itk.GetArrayFromImage(image_itk)
        if dtype in ['short', 'int16', config.DATATYPE_VOXEL_IMG]:    
            img_array = np.array(img_array, dtype=config.DATATYPE_VOXEL_IMG)
        elif dtype in ['unsigned int', 'uint8', config.DATATYPE_VOXEL_MASK]:
            img_array = np.array(img_array, dtype=config.DATATYPE_VOXEL_MASK)
            
        image_sitk = sitk.GetImageFromArray(img_array, isVector=image_itk.GetNumberOfComponentsPerPixel()>1)
        image_sitk.SetOrigin(tuple(image_itk.GetOrigin()))
        image_sitk.SetSpacing(tuple(image_itk.GetSpacing()))
        image_sitk.SetDirection(itk.GetArrayFromMatrix(image_itk.GetDirection()).flatten())
        return image_sitk

    writer = sitk.ImageFileWriter()    
    writer.SetFileName(str(filepath))
    writer.SetUseCompression(compression)
    if 'SimpleITK' not in str(type(img)):
        writer.Execute(convert_itk_to_sitk(img, dtype))
    else:
        writer.Execute(img)

def read_itk(img_url, fixed_ct_skip_slices=None):

    if Path(img_url).exists():
        img = itk.imread(str(img_url), itk.F)

        if fixed_ct_skip_slices is not None:
            img_array = itk.GetArrayFromImage(img) # [D,H,W]
            img_array = img_array[fixed_ct_skip_slices:,:,:]

            img_ = itk.GetImageFromArray(img_array)
            img_.SetOrigin(tuple(img.GetOrigin()))
            img_.SetSpacing(tuple(img.GetSpacing()))
            img_.SetDirection(img.GetDirection())
            img = img_

        return img
    else:
        print (' - [read_itk()] Path does not exist: ', img_url)
        return None

def read_itk_mask(mask_url, fixed_ct_skip_slices=None):

    if Path(mask_url).exists():
        img = itk.imread(str(mask_url), itk.UC)

        if fixed_ct_skip_slices is not None:
            img_array = itk.GetArrayFromImage(img) # [D,H,W]
            img_array = img_array[fixed_ct_skip_slices:,:,:]

            img_ = itk.GetImageFromArray(img_array)
            img_.SetOrigin(tuple(img.GetOrigin()))
            img_.SetSpacing(tuple(img.GetSpacing()))
            img_.SetDirection(img.GetDirection())
            img = img_

        return img

    else:
        print (' - [read_itk()] Path does not exist: ', mask_url)
        return None

def array_to_itk(array, origin, spacing):
    """
    array = [H,W,D]
    origin = [x,y,z]
    spacing = [x,y,z]
    """

    try:
        import itk
        
        img_itk = itk.GetImageFromArray(np.moveaxis(array, [0,1,2], [2,1,0]).copy())
        img_itk.SetOrigin(tuple(origin))
        img_itk.SetSpacing(tuple(spacing))
        # dir = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float64)
        # img_itk.SetDirection(itk.GetMatrixFromArray(dir))

        return img_itk

    except:
        traceback.print_exc()
        pdb.set_trace()

def array_to_sitk(array_input, size=None, origin=None, spacing=None, direction=None, is_vector=False, im_ref=None):
    """
    This function takes an array and converts it into a SimpleITK image.

    Parameters
    ----------
    array_input: numpy
        The numpy array to convert to a SimpleITK image
    size: tuple, optional
        The size of the array
    origin: tuple, optional
        The origin of the data in physical space
    spacing: tuple, optional
        Spacing describes the physical sie of each pixel
    direction: tuple, optional
        A [nxn] matrix passed as a 1D in a row-major form for a nD matrix (n=[2,3]) to infer the orientation of the data
    is_vector: bool, optional
        If isVector is True, then the Image will have a Vector pixel type, and the last dimension of the array will be considered the component index.
    im_ref: sitk image
        An empty image with meta information
    
    Ref: https://github.com/hsokooti/RegNet/blob/46f345d25cd6a1e0ee6f230f64c32bd15b7650d3/functions/image/image_processing.py#L86
    """
    import SimpleITK as sitk
    verbose = False
    
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    if spacing is None:
        spacing = [1, 1, 1]  # the voxel spacing
    if direction is None:
        direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if size is None:
        size = np.array(array_input).shape

    """
    ITK has a GetPixel which takes an ITK Index object as an argument, which is ordered as (x,y,z). 
    This is the convention that SimpleITK's Image class uses for the GetPixel method and slicing operator as well. 
    In numpy, an array is indexed in the opposite order (z,y,x)
    """
    sitk_output = sitk.GetImageFromArray(np.moveaxis(array_input, [0,1,2], [2,1,0]), isVector=is_vector) # np([H,W,D]) -> np([D,W,H]) -> sitk([H,W,D])
    
    if im_ref is None:
        sitk_output.SetOrigin(origin)
        sitk_output.SetSpacing(spacing)
        sitk_output.SetDirection(direction)
    else:
        sitk_output.SetOrigin(im_ref.GetOrigin())
        sitk_output.SetSpacing(im_ref.GetSpacing())
        sitk_output.SetDirection(im_ref.GetDirection())

    return sitk_output

def sitk_to_array(sitk_image):
    array = sitk.GetArrayFromImage(sitk_image)
    array = np.moveaxis(array, [0,1,2], [2,1,0]) # [D,W,H] --> [H,W,D]
    return array

def itk_to_array(sitk_image):
    import itk
    array = itk.GetArrayFromImage(sitk_image)
    array = np.moveaxis(array, [0,1,2], [2,1,0]) # [D,W,H] --> [H,W,D]
    return array

def resampler_sitk(image_sitk, spacing=None, scale=None, im_ref=None, im_ref_size=None, default_pixel_value=0, interpolator=None, dimension=3):
    """
    :param image_sitk: input image
    :param spacing: desired spacing to set
    :param scale: if greater than 1 means downsampling, less than 1 means upsampling
    :param im_ref: if im_ref available, the spacing will be overwritten by the im_ref.GetSpacing()
    :param im_ref_size: in sikt order: x, y, z
    :param default_pixel_value:
    :param interpolator:
    :param dimension:
    :return:
    """

    import math
    import SimpleITK as sitk

    if spacing is None and scale is None:
        raise ValueError('spacing and scale cannot be both None')
    if interpolator is None:
        interpolator = sitk.sitkBSpline # sitk.Linear, sitk.Nearest

    if spacing is None:
        spacing = tuple(i * scale for i in image_sitk.GetSpacing())
        if im_ref_size is None:
            im_ref_size = tuple(round(i / scale) for i in image_sitk.GetSize())

    elif scale is None:
        ratio = [spacing_dim / spacing[i] for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
        if im_ref_size is None:
            im_ref_size = tuple(math.ceil(size_dim * ratio[i]) - 1 for i, size_dim in enumerate(image_sitk.GetSize()))
    else:
        raise ValueError('spacing and scale cannot both have values')

    if im_ref is None:
        im_ref = sitk.Image(im_ref_size, sitk.sitkInt8)
        im_ref.SetOrigin(image_sitk.GetOrigin())
        im_ref.SetDirection(image_sitk.GetDirection())
        im_ref.SetSpacing(spacing)
    

    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampled_sitk = resampler_by_transform(image_sitk, identity, im_ref=im_ref,
                                            default_pixel_value=default_pixel_value,
                                            interpolator=interpolator)
    return resampled_sitk

def resampler_by_transform(im_sitk, dvf_t, im_ref=None, default_pixel_value=0, interpolator=None):
    import SimpleITK as sitk

    if im_ref is None:
        im_ref = sitk.Image(dvf_t.GetDisplacementField().GetSize(), sitk.sitkInt8)
        im_ref.SetOrigin(dvf_t.GetDisplacementField().GetOrigin())
        im_ref.SetSpacing(dvf_t.GetDisplacementField().GetSpacing())
        im_ref.SetDirection(dvf_t.GetDisplacementField().GetDirection())

    if interpolator is None:
        interpolator = sitk.sitkBSpline

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_ref)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(dvf_t)

    # [DEBUG]
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    out_im = resampler.Execute(im_sitk)
    return out_im

def save_as_mha_mask(data_dir, patient_id, voxel_mask, voxel_img_headers):
    
    try:
        voxel_save_folder = Path(data_dir).joinpath(patient_id)
        Path(voxel_save_folder).mkdir(parents=True, exist_ok=True)
        study_id = Path(voxel_save_folder).parts[-1]
        if config.STR_ACCESSION_PREFIX in str(study_id):
            study_id = 'acc_' + str(study_id.split(config.STR_ACCESSION_PREFIX)[-1])
            study_id = study_id.replace('.', '')


        orig_origin = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ORIGIN]
        orig_pixel_spacing = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_PIXEL_SPACING]

        if len(voxel_mask):
            voxel_mask_sitk = array_to_sitk(voxel_mask.astype(config.DATATYPE_VOXEL_MASK)
                                    , origin=orig_origin, spacing=orig_pixel_spacing)
            path_voxel_mask = Path(voxel_save_folder).joinpath(config.FILENAME_MASK_TUMOR_3D.format(study_id))
            sitk.WriteImage(voxel_mask_sitk, str(path_voxel_mask), useCompression=True)
        else:
            print (' - [ERROR][save_as_mha_mask()] Patient: ', patient_id)
    except:
        traceback.print_exc()
        pdb.set_trace()
        pass

def save_as_mha(data_dir, patient_id, voxel_img, voxel_img_headers, voxel_mask
        , voxel_img_reg_dict={}, labelid_midpoint=None
        , resample_spacing=[]):
    
    try:
        """
        Thi function converts the raw numpy data into a SimpleITK image and saves as .mha

        Parameters
        ----------
        data_dir: Path
            The path where you would like to save the data
        patient_id: str
            A reference to the patient
        voxel_img: numpy
            A nD numpy array with [H,W,D] format containing radiodensity data in Hounsfield units
        voxel_img_headers: dict
            A python dictionary containing information on 'origin' and 'pixel_spacing'  
        voxel_mask: numpy
            A nD array with labels on each nD voxel
        resample_save: bool
            A boolean variable to indicate whether the function should resample
        """

        # Step 1 - Original Voxel resolution
        ## Step 1.1 - Create save dir
        voxel_save_folder = Path(data_dir).joinpath(patient_id)
        Path(voxel_save_folder).mkdir(parents=True, exist_ok=True)
        study_id = Path(voxel_save_folder).parts[-1]
        if config.STR_ACCESSION_PREFIX in str(study_id):
            study_id = config.FILEPREFIX_ACCENSION + get_acc_id_from_str(study_id)
        
        ## Step 1.2 - Save img voxel
        orig_origin = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ORIGIN]
        orig_pixel_spacing = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_PIXEL_SPACING]
        voxel_img_sitk = array_to_sitk(voxel_img.astype(config.DATATYPE_VOXEL_IMG)
                            , origin=orig_origin, spacing=orig_pixel_spacing)
        path_voxel_img = Path(voxel_save_folder).joinpath(config.FILENAME_IMG_3D.format(study_id))
        sitk.WriteImage(voxel_img_sitk, str(path_voxel_img), useCompression=True)

        ## Step 1.3 - Save mask voxel
        voxel_mask_sitk = array_to_sitk(voxel_mask.astype(config.DATATYPE_VOXEL_MASK)
                            , origin=orig_origin, spacing=orig_pixel_spacing)
        path_voxel_mask = Path(voxel_save_folder).joinpath(config.FILENAME_MASK_3D.format(study_id))
        sitk.WriteImage(voxel_mask_sitk, str(path_voxel_mask), useCompression=True)

        ## Step 1.4 - Save registration params
        paths_voxel_reg = {}
        for study_id in voxel_img_reg_dict:
            if type(voxel_img_reg_dict[study_id]) == sitk.AffineTransform:
                path_voxel_reg = str(Path(voxel_save_folder).joinpath('{}.tfm'.format(study_id)))
                sitk.WriteTransform(voxel_img_reg_dict[study_id], path_voxel_reg)
                paths_voxel_reg[study_id] = str(path_voxel_reg)

        # Step 2 - Resampled Voxel
        if len(resample_spacing):
            new_spacing = resample_spacing

            ## Step 2.1 - Save resampled img
            voxel_img_sitk_resampled = resampler_sitk(voxel_img_sitk, spacing=new_spacing, interpolator=sitk.sitkBSpline) 
            voxel_img_sitk_resampled = sitk.Cast(voxel_img_sitk_resampled, sitk.sitkInt16)
            path_voxel_img_resampled = Path(voxel_save_folder).joinpath(config.FILENAME_IMG_RESAMPLED_3D.format(study_id))
            sitk.WriteImage(voxel_img_sitk_resampled, str(path_voxel_img_resampled), useCompression=True)
            interpolator_img = 'sitk.sitkBSpline'
            
            ## Step 2.2 - Save resampled mask
            voxel_mask_sitk_resampled = []
            interpolator_mask = ''
            if 0:
                voxel_mask_sitk_resampled = resampler_sitk(voxel_mask_sitk, spacing=new_spacing, interpolator=sitk.sitkNearestNeighbor)
                interpolator_mask = 'sitk.sitkNearestNeighbor'

            elif 1:
                interpolator_mask = 'sitk.sitkLinear'
                new_size = voxel_img_sitk_resampled.GetSize()
                voxel_mask_resampled = np.zeros(new_size)
                for label_id in np.unique(voxel_mask):
                    if label_id != 0:
                        voxel_mask_singlelabel = copy.deepcopy(voxel_mask).astype(config.DATATYPE_VOXEL_MASK)
                        voxel_mask_singlelabel[voxel_mask_singlelabel != label_id] = 0
                        voxel_mask_singlelabel[voxel_mask_singlelabel == label_id] = 1
                        voxel_mask_singlelabel_sitk = array_to_sitk(voxel_mask_singlelabel
                            , origin=orig_origin, spacing=orig_pixel_spacing)
                        voxel_mask_singlelabel_sitk_resampled = resampler_sitk(voxel_mask_singlelabel_sitk, spacing=new_spacing
                                    , interpolator=sitk.sitkLinear) 
                        if 0:
                            voxel_mask_singlelabel_sitk_resampled = sitk.Cast(voxel_mask_singlelabel_sitk_resampled, sitk.sitkUInt8)
                            voxel_mask_singlelabel_array_resampled = sitk_to_array(voxel_mask_singlelabel_sitk_resampled)
                            idxs = np.argwhere(voxel_mask_singlelabel_array_resampled > 0)
                        else:
                            voxel_mask_singlelabel_array_resampled = sitk_to_array(voxel_mask_singlelabel_sitk_resampled)
                            idxs = np.argwhere(voxel_mask_singlelabel_array_resampled >= 0.5)
                        voxel_mask_resampled[idxs[:,0], idxs[:,1], idxs[:,2]] = label_id

                voxel_mask_sitk_resampled = array_to_sitk(voxel_mask_resampled 
                                , origin=orig_origin, spacing=new_spacing)

            voxel_mask_sitk_resampled = sitk.Cast(voxel_mask_sitk_resampled, sitk.sitkUInt8)    
            path_voxel_mask_resampled = Path(voxel_save_folder).joinpath(config.FILENAME_MASK_RESAMPLED_3D.format(study_id))
            sitk.WriteImage(voxel_mask_sitk_resampled, str(path_voxel_mask_resampled), useCompression=True)

            # Step 2.3 - Save voxel info for resampled data
            midpoint_idxs_mean = []
            if labelid_midpoint is not None:
                voxel_mask_resampled_data = sitk_to_array(voxel_mask_sitk_resampled)
                midpoint_idxs = np.argwhere(voxel_mask_resampled_data == labelid_midpoint)
                midpoint_idxs_mean = np.mean(midpoint_idxs, axis=0)

            path_voxel_headers = Path(voxel_save_folder).joinpath(config.FILENAME_VOXEL_INFO)

            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED] = {config.KEYNAME_MEAN_MIDPOINT : midpoint_idxs_mean.tolist()}
            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_PIXEL_SPACING] = voxel_img_sitk_resampled.GetSpacing()
            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_ORIGIN] = voxel_img_sitk_resampled.GetOrigin()
            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_SHAPE] = voxel_img_sitk_resampled.GetSize()
            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.TYPE_VOXEL_ORIGSHAPE] = {
                config.KEYNAME_INTERPOLATOR_IMG: interpolator_img
                , config.KEYNAME_INTERPOLATOR_MASK: interpolator_mask
            }
            if config.KEYNAME_LABEL_OARS in voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE]:
                voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_LABEL_OARS] = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_LABEL_OARS]
            if config.KEYNAME_LABEL_MISSING in voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE]:
                voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_LABEL_MISSING] = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_LABEL_MISSING]

            write_json(voxel_img_headers, path_voxel_headers)   

        ## Step 3 - Save img headers
        path_voxel_headers = Path(voxel_save_folder).joinpath(config.FILENAME_VOXEL_INFO)
        write_json(voxel_img_headers, path_voxel_headers)

        if len(voxel_img_reg_dict):
            return str(path_voxel_img), str(path_voxel_mask), paths_voxel_reg
        else:
            return str(path_voxel_img), str(path_voxel_mask), {}
    
    except:
        print ('\n --------------- [ERROR][save_as_mha()]')
        traceback.print_exc()
        print (' --------------- [ERROR][save_as_mha()]\n')
        pdb.set_trace()

def read_mha(path_file):
    try:
        
        if Path(path_file).exists():
            img_mha = sitk.ReadImage(str(path_file))
            return img_mha
        else:
            print (' - [ERROR][read_mha()] Path issue: path_file: ', path_file)
            pdb.set_trace()
            
    except:
        traceback.print_exc()
        pdb.set_trace()

def write_itk(data, filepath):
    filepath_parent = Path(filepath).parent.absolute()
    filepath_parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(data, str(filepath), compression=True)

def resample_img(path_img, path_new_img, spacing):

    try:
        
        img_sitk      = sitk.ReadImage(str(path_img))
        img_resampled = resampler_sitk(img_sitk, spacing=spacing, interpolator=sitk.sitkBSpline) 
        img_resampled = sitk.Cast(img_resampled, sitk.sitkInt16)

        path_new_parent = Path(path_new_img).parent.absolute()
        Path(path_new_parent).mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(img_resampled, str(path_new_img), useCompression=True)

        return img_resampled

    except:
        traceback.print_exc()

def resample_mask(path_mask, path_new_mask, spacing, size, labels_allowed = []):

    try:
        
        # Step 0 - Init
        mask_array_resampled = np.zeros(size)

        # Step 1 - Read mask
        mask_sitk    = sitk.ReadImage(str(path_mask))
        mask_spacing = mask_sitk.GetSpacing()
        mask_origin  = mask_sitk.GetOrigin()
        mask_array   = sitk_to_array(mask_sitk)
        
        # Step 2 - Loop over mask labels
        for label_id in np.unique(mask_array):
            if label_id != 0 and label_id in labels_allowed:
                mask_array_label = np.array(mask_array, copy=True)
                mask_array_label[mask_array_label != label_id] = 0
                mask_array_label[mask_array_label == label_id] = 1
                mask_sitk_label = array_to_sitk(mask_array_label, origin=mask_origin, spacing=mask_spacing)
                mask_sitk_label_resampled = resampler_sitk(mask_sitk_label, spacing=spacing, interpolator=sitk.sitkLinear)
                mask_array_label_resampled = sitk_to_array(mask_sitk_label_resampled)
                idxs = np.argwhere(mask_array_label_resampled >= 0.5)
                mask_array_resampled[idxs[:,0], idxs[:,1], idxs[:,2]] = label_id
        
        mask_sitk_resampled = array_to_sitk(mask_array_resampled , origin=mask_origin, spacing=spacing)            
        mask_sitk_resampled = sitk.Cast(mask_sitk_resampled, sitk.sitkUInt8)

        path_new_parent = Path(path_new_mask).parent.absolute()
        Path(path_new_parent).mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(mask_sitk_resampled, str(path_new_mask), useCompression=True)

        return mask_sitk_resampled

    except:
        traceback.print_exc()

############################################################
#                    3D VOXEL RELATED                      #
############################################################

def split_into_overlapping_grids(len_total, len_grid, len_overlap, res_type='boundary'):
  res_range = []
  res_boundary = []

  A = np.arange(len_total)
  l_start = 0
  l_end = len_grid
  while(l_end < len(A)):
    res_range.append(np.arange(l_start, l_end))
    res_boundary.append([l_start, l_end])
    l_start = l_start + len_grid - len_overlap
    l_end = l_start + len_grid
  
  res_range.append(np.arange(len(A)-len_grid, len(A)))
  res_boundary.append([len(A)-len_grid, len(A)])
  if res_type == 'boundary':
    return res_boundary
  elif res_type == 'range':
    return res_range

def extract_numpy_from_dcm(patient_dir, skip_slices=None):
    """
    Given the path of the folder containing the .dcm files, this function extracts 
    a numpy array by combining them

    Parameters
    ----------
    patient_dir: Path
        The path of the folder containing the .dcm files
    """
    slices = []
    voxel_img_data = []
    try:
        import pydicom
        from pathlib import Path

        if Path(patient_dir).exists():
            for path_ct in Path(patient_dir).iterdir():
                try:
                    ds = pydicom.filereader.dcmread(path_ct)
                    slices.append(ds)
                except:
                    pass
            
            if len(slices):
                slices = list(filter(lambda x: 'ImagePositionPatient' in x, slices))
                slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) # inferior to superior
                slices_data = []
                for s_id, s in enumerate(slices):
                    try:
                        if skip_slices is not None:
                            if s_id < skip_slices:
                                continue
                        slices_data.append(s.pixel_array.T)
                    except:
                        pass
                
                if len(slices_data):
                    voxel_img_data = np.stack(slices_data, axis=-1) # [row, col, plane], [H,W,D]

    except:
        pass
        traceback.print_exc()
        # pdb.set_trace()
    
    return slices, voxel_img_data

def perform_hu(voxel_img, intercept, slope):
    """
    Rescale Intercept != 0, Rescale Slope != 1 or Dose Grid Scaling != 1.
    The pixel data has not been transformed according to these values.
    Consider using the module ApplyDicomPixelModifiers after
    importing the volume to transform the image data appropriately.
    """
    try:
        import copy

        slope = np.float(slope)
        intercept = np.float(intercept)
        voxel_img_hu = copy.deepcopy(voxel_img).astype(np.float64)

        # Convert to Hounsfield units (HU)    
        if slope != 1:
            voxel_img_hu = slope * voxel_img_hu
            
        voxel_img_hu += intercept

        return voxel_img_hu.astype(config.DATATYPE_VOXEL_IMG)
    except:
        traceback.print_exc()
        pdb.set_trace()

def print_final_message():
    print ('')
    print (' - Note: You can view the 3D data in visualizers like MeVisLab or 3DSlicer')
    print (' - Note: Raw Voxel Data ({}/{}) is in Hounsfield units (HU) with int16 datatype'.format(config.FILENAME_IMG_3D, config.FILENAME_IMG_RESAMPLED_3D))
    print ('')

def volumize_ct(patient_ct_dir, skip_slices=None):
    """
    Params
    ------
    patient_ct_dir: Path - contains .dcm files for CT scans
    """
    voxel_ct_data = []
    voxel_ct_headers = {}

    try:
        
        if Path(patient_ct_dir).exists():

            # Step1 - Accumulate all slices and sort
            slices, voxel_ct_data = extract_numpy_from_dcm(Path(patient_ct_dir), skip_slices=skip_slices)

            # Step 2 - Get parameters related to the 3D scan
            if len(voxel_ct_data):
                slice_thickness = 0
                if slices[0].SliceThickness < 0.01:
                    slice_thickness = float(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
                else:
                    slice_thickness = float(slices[0].SliceThickness)

                voxel_ct_headers[config.KEYNAME_PIXEL_SPACING] = np.array(slices[0].PixelSpacing).tolist() + [slice_thickness]
                voxel_ct_headers[config.KEYNAME_ORIGIN]        = np.array(slices[0].ImagePositionPatient).tolist()
                voxel_ct_headers[config.KEYNAME_SHAPE]         = voxel_ct_data.shape
                voxel_ct_headers[config.KEYNAME_ZVALS]         = [round(s.ImagePositionPatient[2]) for s in slices]
                voxel_ct_headers[config.KEYNAME_INTERCEPT]     = slices[0].RescaleIntercept
                voxel_ct_headers[config.KEYNAME_SLOPE]         = slices[0].RescaleSlope

                # Step 3 - Postprocess the 3D voxel data (data in HU + reso=[config.VOXEL_DATA_RESO]) 
                voxel_ct_data = perform_hu(voxel_ct_data, intercept=slices[0].RescaleIntercept, slope=slices[0].RescaleSlope)
            
            else:
                print (' - [ERROR][volumize_ct()] Error with numpy extraction of CT volume: ', patient_ct_dir)
            
        else:
            print (' - [ERROR][volumize_ct()] Path issue: ', patient_ct_dir)
    except:
        traceback.print_exc()
        pdb.set_trace()
        
    return voxel_ct_data, voxel_ct_headers

############################################################
#             3D VOXEL RELATED - RTSTRUCT                  #
############################################################

def get_self_label_id(label_name_orig, RAW_TO_SELF_LABEL_MAPPING, LABEL_MAP):
        
    try:
        if label_name_orig in RAW_TO_SELF_LABEL_MAPPING:
            label_name_self = RAW_TO_SELF_LABEL_MAPPING[label_name_orig] 
            if label_name_self in LABEL_MAP:
                return LABEL_MAP[label_name_self], label_name_self
            else:
                return 0, ''
        else:
            return 0, ''
    except:
        traceback.print_exc()
        print (' - [ERROR][get_global_label_id()] label_name_orig: ', label_name_orig)
        # pdb.set_trace()
        return 0, ''

def extract_contours_from_rtstruct(rtstruct_ds, RAW_TO_SELF_LABEL_MAPPING=None, LABEL_MAP=None, verbose=False):
    """
    Params
    ------
    rtstruct_ds: pydicom.dataset.FileDataset
        - .StructureSetROISequence: contains identifying info on all contours
        - .ROIContourSequence     : contains a set of contours for particular ROI
    """

    # Step 0 - Init
    contours     = []
    labels_debug = {}

    # Step 1 - Loop and extract all different contours
    for i in range(len(rtstruct_ds.ROIContourSequence)):

        try:

            # Step 1.1 - Get contour 
            contour = {}
            contour['name']     = str(rtstruct_ds.StructureSetROISequence[i].ROIName)
            contour['color']    = list(rtstruct_ds.ROIContourSequence[i].ROIDisplayColor)
            contour['contours'] = [s.ContourData for s in rtstruct_ds.ROIContourSequence[i].ContourSequence]
            
            if RAW_TO_SELF_LABEL_MAPPING is None and LABEL_MAP is None:
                contour['number']        = rtstruct_ds.ROIContourSequence[i].ReferencedROINumber
                assert contour['number'] == rtstruct_ds.StructureSetROISequence[i].ROINumber
            elif RAW_TO_SELF_LABEL_MAPPING is None and LABEL_MAP is not None:
                contour['number'] = int(LABEL_MAP.get(contour['name'], -1))
            else:
                label_id, _       = get_self_label_id(contour['name'], RAW_TO_SELF_LABEL_MAPPING, LABEL_MAP)
                contour['number'] = label_id
            
            if verbose: print (' - name: ', contour['name'],  LABEL_MAP.get(contour['name'], -1))
            
            # Step 1.2 - Keep or not condition
            if contour['number'] > 0:
                contours.append(contour)

            # Step 1.3 - Some debugging
            labels_debug[contour['name']] = {'id': len(labels_debug) + 1}

        except:
            if verbose: print ('\n ---------- [ERROR][utils.extract_contours_from_rtstruct()] name: ', rtstruct_ds.StructureSetROISequence[i].ROIName)

    # Step 2 - Order your contours
    if len(contours):
        contours = list(sorted(contours, key = lambda obj: obj['number']))
    
    return contours, labels_debug

def process_contours(contour_obj_list, params, voxel_mask_data, special_processing_ids = []):
    """
    Goal: Convert contours to voxel mask
    Params
    ------
    contour_obj_list: [{'name':'', 'contours': [[], [], ... ,[]]}, {}, ..., {}]
    special_processing_ids: For donut shaped contours
    """
    import skimage
    import skimage.draw

    # Step 1 - Get some position and spacing params 
    z         = params[config.KEYNAME_ZVALS]
    pos_r     = params[config.KEYNAME_ORIGIN][1]
    spacing_r = params[config.KEYNAME_PIXEL_SPACING][1]
    pos_c     = params[config.KEYNAME_ORIGIN][0]
    spacing_c = params[config.KEYNAME_PIXEL_SPACING][0]
    shape     = params[config.KEYNAME_SHAPE]

    # Step 2 - Loop over contour objects
    for contour_obj in contour_obj_list:

        try:

            class_id = int(contour_obj['number'])

            if class_id not in special_processing_ids:

                # Step 2.1 - Pick a contour for a particular ROI
                for c_id, contour in enumerate(contour_obj['contours']):
                    coords = np.array(contour).reshape((-1, 3))
                    if len(coords) > 1:

                        # Step 2.2 - Get the z-index of a particular z-value
                        assert np.amax(np.abs(np.diff(coords[:, 2]))) == 0
                        z_index = z.index(pydicom.valuerep.DSfloat(float(round(coords[0, 2]))))
                        
                        # Step 2.4 - Polygonize
                        rows = (coords[:, 1] - pos_r) / spacing_r  #pixel_idx = f(real_world_idx, ct_resolution)
                        cols = (coords[:, 0] - pos_c) / spacing_c
                        if 1:
                            rr, cc = skimage.draw.polygon(rows, cols)  # rr --> y-axis, cc --> x-axis                      
                            voxel_mask_data[cc, rr, z_index] = class_id
                        else:
                            contour_mask = skimage.draw.polygon2mask(voxel_mask_data.shape[:2], np.hstack((rows[np.newaxis].T, cols[np.newaxis].T)))
                            contour_idxs = np.argwhere(contour_mask > 0)
                            rr, cc = contour_idxs[:,0], contour_idxs[:,1]
                            voxel_mask_data[cc, rr, z_index] = class_id
                        
                        # Step 2.99 - Debug
                        # f,axarr = plt.subplots(1,2); axarr[0].scatter(cols, rows); axarr[0].invert_yaxis(); axarr[1].imshow(contour_mask);plt.suptitle('Z={:f}'.format(contour[0, 2])); plt.show()
                        
            else:

                # Step 2.1 - Gather all contours for a particular ROI
                contours_all = []
                for contour in contour_obj['contours']: contours_all.extend(contour)
                contours_all = np.array(contours_all).reshape((-1,3))
                
                # Step 2.2 - Split contours on the basis of z-value
                for c_id, contour in enumerate([contours_all[contours_all[:,2]==z_pos] for z_pos in np.unique(contours_all[:,2])]):
                    
                    # Step 2.3 - Get the z-index of a particular z-value
                    assert np.amax(np.abs(np.diff(contour[:, 2]))) == 0
                    z_index = z.index(pydicom.valuerep.DSfloat(float(round(contour[0, 2]))))

                    # Step 2.4 - Polygonize
                    rows = (contour[:, 1] - pos_r) / spacing_r  # pixel_idx = f(real_world_idx, ct_resolution)
                    cols = (contour[:, 0] - pos_c) / spacing_c
                    rr, cc = skimage.draw.polygon(rows, cols)
                    voxel_mask_data[cc, rr, z_index] = class_id

                    # Step 2.99 - Debug
                    # if class_id == 8 and c_id > 3 and c_id < 7:  # larynx
                    #     import matplotlib.pyplot as plt
                    #     f,axarr = plt.subplots(1,2); axarr[0].scatter(cols, rows); axarr[0].invert_yaxis(); axarr[1].scatter(cc, rr);axarr[1].invert_yaxis();plt.suptitle('Z={:f}'.format(contour[0, 2])); plt.show()
                    #     pdb.set_trace()

        except:
            print (' --- [ERROR][utils.process_contours()] contour-number:', contour_obj['number'], ' || contour-label:', contour_obj['name'])
            traceback.print_exc()
    
    return voxel_mask_data

def volumize_rtstruct(patient_rtstruct_path, params, params_labelinfo):
    """
    This function takes a .dcm file (modality=RTSTRUCT) in a folder and converts it into a numpy mask

    Parameters
    ----------
    patient_rtstruct_path: Path
        path to the .dcm file (modality=RTSTRUCT)
    params: dictionary
        A python dictionary containing the following keys - ['pixel_spacing', 'origin', 'z_vals'].
        'z_vals' is a list of all the depth values of the raw .dcm slices
    """

    # Step 0 - Init
    mask_data_oars      = []
    mask_data_external  = []
    mask_headers        = {config.KEYNAME_LABEL_OARS:[], config.KEYNAME_LABEL_EXTERNAL:[]}
    LABEL_MAP_DCM      = params_labelinfo.get(config.KEY_LABELMAP_DCM, None)
    LABEL_MAP_FULL     = params_labelinfo.get(config.KEY_LABEL_MAP_FULL, None)
    LABEL_IDS_SPECIAL  = params_labelinfo.get(config.KEY_ARTFORCE_DONUTSHAPED_IDS, None)
    LABEL_MAP_EXTERNAL = params_labelinfo.get(config.KEY_LABEL_MAP_EXTERNAL, None)

    try:

        ds  = pydicom.filereader.dcmread(patient_rtstruct_path)
            
        if ds.Modality == config.MODALITY_RTSTRUCT:
            
            if config.KEYNAME_SHAPE in params:
                
                # Step 1 - Extract all different contours (for OARs)
                if LABEL_MAP_DCM is not None or LABEL_MAP_FULL is not None:
                    contours_oars, _ = extract_contours_from_rtstruct(ds, LABEL_MAP_DCM, LABEL_MAP_FULL)
                    if len(contours_oars):
                        mask_headers[config.KEYNAME_LABEL_OARS] = [contour['name'] for contour in contours_oars]
                        mask_data_oars = np.zeros(params[config.KEYNAME_SHAPE], dtype=np.uint8)
                        mask_data_oars = process_contours(contours_oars, params, mask_data_oars, special_processing_ids=LABEL_IDS_SPECIAL)
                    else:
                        print (' - [ERROR][volumize_rtstruct()] Len of OAR contours is 0')
                
                # Step 2 - Extract contour for "External"
                if LABEL_MAP_EXTERNAL is not None:
                    contours_external, _ = extract_contours_from_rtstruct(ds, LABEL_MAP_DCM, LABEL_MAP_EXTERNAL)
                    if len(contours_external):
                        mask_headers[config.KEYNAME_LABEL_EXTERNAL] = [contour['name'] for contour in contours_external]
                        mask_data_external = np.zeros(params[config.KEYNAME_SHAPE], dtype=np.uint8)
                        mask_data_external = process_contours(contours_external, params, mask_data_external)
                    else:
                        print (' - [ERROR][volumize_rtstruct()] Len of External contours is 0')

            else:
                print (' - [ERROR][volumize_rtstruct()] Issue with voxel params: ', params)

        else:
            print (' - [ERROR][volumize_rtstruct()] Could not capture RTSTRUCT file')
    
    except:
        traceback.print_exc()
        pdb.set_trace()

    return mask_data_oars, mask_data_external, mask_headers

############################################################
#                SAVING/READING RELATED                    #
############################################################

def save_csv(filepath, data_array):
    Path(filepath).parent.absolute().mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, data_array, fmt='%s')

def read_csv(filepath):
    data = np.loadtxt(filepath, dtype='str')
    return data

def write_json(json_data, json_filepath):

    Path(json_filepath).parent.absolute().mkdir(parents=True, exist_ok=True)

    with open(str(json_filepath), 'w') as fp:
        json.dump(json_data, fp, indent=4, cls=NpEncoder)

def read_json(json_filepath, verbose=True):

    if Path(json_filepath).exists():
        with open(str(json_filepath), 'r') as fp:
            data = json.load(fp)
            return data
    else:
        if verbose: print (' - [ERROR][read_json()] json_filepath does not exist: ', json_filepath)
        return None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def write_nrrd(filepath, data, spacing):
    nrrd_headers = {'space':'left-posterior-superior', 'kinds': ['domain', 'domain', 'domain'], 'encoding':'gzip'}
    space_directions = np.zeros((3,3), dtype=np.float32)
    space_directions[[0,1,2],[0,1,2]] = np.array(spacing)
    nrrd_headers['space directions'] = space_directions

    import nrrd
    nrrd.write(str(filepath), data, nrrd_headers)

############################################################
#                           RANDOM                         #
############################################################
def get_name_patient_study_id(meta):
    try:
        meta = np.array(meta)
        meta = str(meta.astype(str))

        meta_split = meta.split('-')
        name = None
        patient_id = None
        study_id = None

        if len(meta_split) == 2:
            name = meta_split[0]
            patient_id = meta_split[1]
        elif len(meta_split) == 3:
            name = meta_split[0]
            patient_id = meta_split[1]
            study_id = meta_split[2]
        
        return name, patient_id, study_id
    
    except:
        traceback.print_exc()
        pdb.set_trace()

def get_numbers(string):
    return ''.join([s for s in string if s.isdigit()])

def create_folds(idxs, folds=4):
    
    # Step 0 - Init
    res = {}
    idxs = np.array(idxs)

    try:
        for fold in range(folds):
            print (' - [create_crossvalfolds()] fold: ', fold)
            val_idxs   = list(np.random.choice(np.arange(len((idxs))), size=int(len(idxs)/folds), replace=False))
            train_idxs = list( set(np.arange(len(idxs))) - set(val_idxs) )
            val_patients   = idxs[val_idxs]
            train_patients = idxs[train_idxs]

            res[fold+1] = {config.MODE_TRAIN: train_patients, config.MODE_VAL: val_patients}

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return res

############################################################
#                    DEBUG RELATED                         #
############################################################

def benchmark_model(model_time):
    time.sleep(model_time)

def benchmark(dataset_generator, model_time=0.05):

    import psutil
    import humanize
    import pynvml 
    pynvml.nvmlInit()

    device_id = pynvml.nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process(os.getpid())

    print ('\n - [benchmark()]')
    t99 = time.time()
    steps = 0
    t0 = time.time()
    for X,_,meta1,meta2 in dataset_generator:
        t1 = time.time()
        benchmark_model(model_time)
        t2 = time.time()
        steps += 1
        # print (' - Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s', '(',X.shape,'), (',meta2.numpy(),')')
        print (' - Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s' \
                # , '(', humanize.naturalsize( process.memory_info().rss),'), ' \
                # , '(', '%.4f' % (pynvml.nvmlDeviceGetMemoryInfo(device_id).used/1024/1024/1000),'GB), ' \
                , '(', meta2.numpy(),')'
                # , np.sum(meta1[:,-9:].numpy(), axis=1)
                # , meta1[:,1:4].numpy()
            )
        t0 = time.time()
    t99 = time.time() - t99
    print ('\n - Total steps: ', steps)
    print (' - Total time for dataset: ', round(t99,2), 's')

def benchmark2(dataset_generator, model_time=0.05):

    import psutil
    import humanize
    import pynvml 
    pynvml.nvmlInit()

    device_id = pynvml.nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process(os.getpid())

    print ('\n - [benchmark2()]')
    t99 = time.time()
    steps = 0
    t0 = time.time()
    for X_moving,_,_,_,meta1,meta2 in dataset_generator:
        t1 = time.time()
        benchmark_model(model_time)
        t2 = time.time()
        steps += 1
        # print (' - Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s', '(',X_moving.shape,'), (',meta2.numpy(),')')
        print (' - Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s' \
                , '(', humanize.naturalsize( process.memory_info().rss),'), ' \
                , '(', '%.4f' % (pynvml.nvmlDeviceGetMemoryInfo(device_id).used/1024/1024/1000),'GB), ' \
                # , '(', meta2.numpy(),')'
                # , np.sum(meta1[:,-9:].numpy(), axis=1)
                # , meta1[:,1:4].numpy()
            )
        t0 = time.time()
    t99 = time.time() - t99
    print ('\n - Total steps: ', steps)
    print (' - Total time for dataset: ', round(t99,2), 's')

def print_debug_header():
    print (' ============================================== ')
    print ('                   DEBUG ')
    print (' ============================================== ')

def get_memory(pid):
    try:
        process = psutil.Process(pid)
        return humanize.naturalsize(process.memory_info().rss)
    except:
        return '-1'

############################################################
#                   DATALOADER RELATED                     #
############################################################

def get_dataloader_3D_train(data_dir, dir_type=['train', 'train_additional']
                    , dimension=3, grid=True, crop_init=True, resampled=True, mask_type=config.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False, random_grid=True
                    , parallel_calls=None, deterministic=False
                    , patient_shuffle=True
                    , centred_dataloader_prob=0.0
                    , debug=False
                    , pregridnorm=True):

    debug = False
    datasets = []
    
    # Dataset 1
    for dir_type_ in dir_type:

        # Step 1 - Get dataset class
        dataset_han_miccai2015 = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type_
                    , dimension=dimension, grid=grid, crop_init=crop_init, resampled=resampled, mask_type=mask_type
                    , transforms=transforms, filter_grid=filter_grid, random_grid=random_grid
                    , parallel_calls=parallel_calls, deterministic=deterministic
                    , patient_shuffle=patient_shuffle
                    , centred_dataloader_prob=centred_dataloader_prob
                    , debug=debug
                    , pregridnorm=pregridnorm)

        # Step 2 - Training transforms
        x_shape_w = dataset_han_miccai2015.w_grid
        x_shape_h = dataset_han_miccai2015.h_grid
        x_shape_d = dataset_han_miccai2015.d_grid
        label_map = dataset_han_miccai2015.LABEL_MAP
        transforms = [
                    aug.Rotate3DSmall(label_map, mask_type)
                    , aug.Deform2Punt5D((x_shape_h, x_shape_w, x_shape_d), label_map, grid_points=50, stddev=4, div_factor=2, debug=False)
                    , aug.Translate(label_map, translations=[40,40])
                    , aug.Noise(x_shape=(x_shape_h, x_shape_w, x_shape_d, 1), mean=0.0, std=0.1)
                ]
        dataset_han_miccai2015.transforms = transforms 

        # Step 3 - Training filters for background-only grids
        if filter_grid:
            dataset_han_miccai2015.filter = aug.FilterByMask(len(dataset_han_miccai2015.LABEL_MAP), dataset_han_miccai2015.SAMPLER_PERC)

        # Step 4 - Append to list
        datasets.append(dataset_han_miccai2015)
    
    dataset = ZipDataset(datasets)
    return dataset

def get_dataloader_3D_train_eval(data_dir, dir_type=['train_train_additional']
                    , dimension=3, grid=True, crop_init=True, resampled=True, mask_type=config.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False, random_grid=False
                    , parallel_calls=None, deterministic=True
                    , patient_shuffle=False
                    , centred_dataloader_prob=0.0
                    , debug=False
                    , pregridnorm=True):
    
    datasets = []
    
    # Dataset 1
    for dir_type_ in dir_type:

        # Step 1 - Get dataset class
        dataset_han_miccai2015 = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type_
                    , dimension=dimension, grid=grid, crop_init=crop_init, resampled=resampled, mask_type=mask_type
                    , transforms=transforms, filter_grid=filter_grid, random_grid=random_grid
                    , parallel_calls=parallel_calls, deterministic=deterministic
                    , patient_shuffle=patient_shuffle
                    , centred_dataloader_prob=centred_dataloader_prob
                    , debug=debug
                    , pregridnorm=pregridnorm)

        # Step 2 - Training transforms
        # None
        
        # Step 3 - Training filters for background-only grids
        # None

        # Step 4 - Append to list
        datasets.append(dataset_han_miccai2015)
    
    dataset = ZipDataset(datasets)
    return dataset

def get_dataloader_3D_test_eval(data_dir, dir_type=['test_offsite']
                    , dimension=3, grid=True, crop_init=True, resampled=True, mask_type=config.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False, random_grid=False
                    , parallel_calls=None, deterministic=True
                    , patient_shuffle=False
                    , debug=False
                    , pregridnorm=True):
    
    datasets = []

    # Dataset 1
    for dir_type_ in dir_type:

        # Step 1 - Get dataset class
        dataset_han_miccai2015 = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type_
                    , dimension=dimension, grid=grid, crop_init=crop_init, resampled=resampled, mask_type=mask_type
                    , transforms=transforms, filter_grid=filter_grid, random_grid=random_grid
                    , parallel_calls=parallel_calls, deterministic=deterministic
                    , patient_shuffle=patient_shuffle
                    , debug=debug
                    , pregridnorm=pregridnorm)

        # Step 2 - Testing transforms
        # None

        # Step 3 - Testing filters for background-only grids
        # None

        # Step 4 - Append to list
        datasets.append(dataset_han_miccai2015)

    dataset = ZipDataset(datasets)
    return dataset

def get_dataloader_deepmindtcia(data_dir
                , dir_type=[config.DATALOADER_DEEPMINDTCIA_TEST]
                , annotator_type=[config.DATALOADER_DEEPMINDTCIA_ONC]
                , grid=True, crop_init=True, resampled=True, mask_type=config.MASK_TYPE_ONEHOT
                , transforms=[], filter_grid=False, random_grid=False, pregridnorm=True
                , parallel_calls=None, deterministic=False
                , patient_shuffle=True
                , centred_dataloader_prob = 0.0
                , debug=False):

    datasets = []
    
    # Dataset 1
    for dir_type_ in dir_type:

        for anno_type_ in annotator_type:

            # Step 1 - Get dataset class
            dataset_han_deepmindtcia = HaNDeepMindTCIADataset(data_dir=data_dir, dir_type=dir_type_, annotator_type=anno_type_
                        , grid=grid, crop_init=crop_init, resampled=resampled, mask_type=mask_type, pregridnorm=pregridnorm
                        , transforms=transforms, filter_grid=filter_grid, random_grid=random_grid
                        , parallel_calls=parallel_calls, deterministic=deterministic
                        , patient_shuffle=patient_shuffle
                        , centred_dataloader_prob=centred_dataloader_prob
                        , debug=debug)
        
            # Step 2 - Append to list
            datasets.append(dataset_han_deepmindtcia)
    
    dataset = ZipDataset(datasets)
    return dataset

