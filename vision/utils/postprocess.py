import os
import shutil
from logger import MyLogger
from logging import DEBUG
from dotenv import load_dotenv
load_dotenv()


my_logger = MyLogger("Postprocess", level=DEBUG)

DICOM_ROOT = os.environ["DICOM_ROOT"]
NIFTI_ROOT = os.environ["NIFTI_ROOT"]
SC_FOLDER = os.environ["SC_FOLDER"]


def clean_patient(patient):

    try:
        # RAW
        my_logger.info("{} dicoms files are removing from raw folder"
                       .format(patient))
        # shutil.rmtree(os.path.join(DICOM_ROOT, patient))

        # NIFTI
        my_logger.info("{} dicoms files are removing from nifti folder"
                       .format(patient))
        shutil.rmtree(os.path.join(NIFTI_ROOT, patient))

        # SC
        my_logger.info("{} dicoms files are removing from raw folder"
                       .format(patient))
        shutil.rmtree(os.path.join(SC_FOLDER, patient))

    except Exception:
        my_logger.error("Something goes wrong when {} patient cleans"
                        .format(patient), exc_info=True)