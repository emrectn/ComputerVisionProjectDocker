import os
import pydicom
import numpy as np
from PIL import Image
import dicom2nifti
from logger import MyLogger
from logging import DEBUG
import magic
from utils.custom_exception import FolderStructureError
from utils.visualize import overlay_segmentation
import datetime, time

my_logger = MyLogger("Preprocess", level=DEBUG)

SC_FOLDER = os.environ["SC_FOLDER"]


def isDicom(path):
    check = magic.from_file(path, mime=True)
    check = check.replace("\\", "/").rsplit("/", 1)[-1]
    return True if check == 'dicom' else False


def vol_rgb(vol):
    vol = np.clip(vol, -1250, 250)
    # Scale volume to greyscale range
    vol_scaled = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    vol_greyscale = np.around(vol_scaled * 255, decimals=0).astype(np.uint8)
    # Convert volume to RGB
    vol_rgb = np.stack([vol_greyscale, vol_greyscale, vol_greyscale], axis=-1)
    return vol_rgb


def export_results(predictions, data_io, patient, patient_info):
    current_date = str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    destination = os.path.join(os.environ["DEST"], patient)

    # RESULT

    data = dict()
    data["patient_id"] = patient
    data["results"] = []
    data["status"] = "DONE"
    data["isCovid"] = None
    data["date"] = current_date[:10]
    data["time"] = current_date[11:]

    sample = data_io.sample_loader(patient, load_seg=False, load_pred=False)
    image = sample.img_data
    vol = np.squeeze(image, axis=-1)
    vol_pred = overlay_segmentation(vol, predictions[0])

    # patient folder
    os.makedirs("{}".format(destination), exist_ok=True)
    # patient original folder
    os.makedirs("{}/original".format(destination), exist_ok=True)
    # patient result folder
    os.makedirs("{}/result".format(destination), exist_ok=True)

    rgb_originial = vol_rgb(vol)

    symptom_count = 0

    result_images = []

    for i in range(vol_pred.shape[2]):

        index = vol_pred.shape[2]-i-1
        im = Image.fromarray(vol_pred[:,:,index])
        im = im.rotate(90, expand=True)

        result_images.append(im)

    for i, im in enumerate(result_images):
        result = dict()
        result["instanceNumber"] = i
        index = vol_pred.shape[2]-i-1
        if i  == 0:
            my_logger.info("Results are converting to JPEG")
        result["result_path"] = "{}/result/{}.png".format(destination, i)
        im.save(result["result_path"])

        im_original = Image.fromarray(rgb_originial[:,:,index])   
        im_original = im_original.rotate(90, expand=True)
        im_original = im_original.convert("L")
        result["original_path"] = "{}/original/{}.png".format(destination,  i)
        im_original.save(result["original_path"])

        # Covid Volume
        covid_volume = np.count_nonzero(predictions[0][:, :, index] == 3)

        if covid_volume > 9:
            symptom_count += 1
            result["isCovid"] = 1
        else:
            result["isCovid"] = 0

        result["covid_volume"] = covid_volume
        # append main result
        data["results"].append(result)

    data["symptom_count"] = symptom_count
    return data


def dcm2nifti(patient):

    src = os.path.join(os.environ["DICOM_ROOT"], patient)
    if not os.path.isdir(src):
        raise(FolderStructureError("Source: {} is Not Folder".format(src)))

    for root, dirs, files in os.walk(src):

        if len(dirs) > 1:
            raise(FolderStructureError("Wrong Folder Structure"))

        elif len(dirs) > 0:
            continue

        print("Dosyadaki DICOMLAR Kontrol İşleniyor")

        for index, f in enumerate(files):

            dicom_path = os.path.join(root, f)
            if isDicom(dicom_path):

                patient_info = get_patient_info(dicom_path)
                dicom_source = root

                break

    dest = os.path.join(os.environ["NIFTI_ROOT"], patient)
    os.makedirs(dest, exist_ok=True)
    my_logger.info(" Dicom series converting to nifti in {} folder".format(dicom_source))
    dicom2nifti.convert_directory(dicom_source, dest, compression=True, reorient=True)

    gz_file = os.listdir(dest)[0]
    os.rename(os.path.join(dest, gz_file), os.path.join(dest,"imaging.nii.gz"))
    return patient_info


def get_patient_info(dicom_path):
    # REFACTOR: check if Image folder here
    dicom = pydicom.dcmread(dicom_path)

    if not dicom:
        raise(FolderStructureError("Couldnt find any dicom in this folder"))

    patient_info = dict()

    necessary_att = ['InstitutionName', 'InstitutionAddress', 'PatientBirthDate',
                     'PatientAge', 'PatientName', 'StudyID', 'StudyDate', 'PatientID','StudyInstanceUID','AccessionNumber']

    for att in necessary_att:
        if att in dicom:
            patient_info[att] = str(dicom[att].value)
        else:
            my_logger.error(" Not Found Attribute: {}".format(att))

    tmp = [patient_info['InstitutionName'].lower(), patient_info['InstitutionAddress'].lower()]

    if any("ankara" in inst for inst in tmp):
        patient_info["Institution"] = "ANKARA"

    elif any("icerenkoy" in inst for inst in tmp):
        patient_info["Institution"] = "ICERENKOY"

    return patient_info
