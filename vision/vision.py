
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn import Data_IO, Preprocessor, Data_Augmentation, Neural_Network
from miscnn.processing.subfunctions import Normalization, Clipping, Resampling
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import tversky_crossentropy, dice_soft, dice_crossentropy, tversky_loss
import os
from dotenv import load_dotenv

# db
from db import mongo
from logger import MyLogger
from datetime import datetime
from logging import DEBUG
from utils.custom_exception import FolderStructureError

from utils.preprocess import export_results, dcm2nifti
from utils.postprocess import clean_patient

my_logger = MyLogger("vision", level=DEBUG)
load_dotenv()

DICOM_ROOT = os.environ["DICOM_ROOT"]
DEST = os.environ["DEST"]
MODEL_FILE = os.environ["MODEL_FILE"]
NIFTI_ROOT = os.environ["NIFTI_ROOT"]


def initalize():
    os.makedirs(os.environ["DEST"], exist_ok=True)
    os.makedirs(os.environ["DICOM_ROOT"], exist_ok=True)
    os.makedirs(os.environ["NIFTI_ROOT"], exist_ok=True)


def get_model(model_file=MODEL_FILE):

    # Initialize Data IO Interface for NIfTI data
    # We are using 4 classes due to [background, lung_left, lung_right, covid-19]
    interface = NIFTI_interface(channels=1, classes=4)

    # Create Data IO object to load and write samples in the file structure
    data_io = Data_IO(interface, input_path="data/nifti", delete_batchDir=False)

    # Access all available samples in our file structure
    sample_list = data_io.get_indiceslist()
    sample_list.sort()

    # Create and configure the Data Augmentation class
    data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
                                 elastic_deform=True, mirror=True,
                                 brightness=True, contrast=True, gamma=True,
                                 gaussian_noise=True)

    # Create a clipping Subfunction to the lung window of CTs (-1250 and 250)
    sf_clipping = Clipping(min=-1250, max=250)
    # Create a pixel value normalization Subfunction to scale between 0-255
    sf_normalize = Normalization(mode="grayscale")
    # Create a resampling Subfunction to voxel spacing 1.58 x 1.58 x 2.70
    sf_resample = Resampling((1.58, 1.58, 2.70))
    # Create a pixel value normalization Subfunction for z-score scaling
    sf_zscore = Normalization(mode="z-score")

    # Assemble Subfunction classes into a list
    sf = [sf_clipping, sf_normalize, sf_resample, sf_zscore]

    # Create and configure the Preprocessor class
    pp = Preprocessor(data_io, data_aug=data_aug, batch_size=2, subfunctions=sf,
                      prepare_subfunctions=True, prepare_batches=False,
                      analysis="patchwise-crop", patch_shape=(160, 160, 80))
    # Adjust the patch overlap for predictions
    pp.patchwise_overlap = (80, 80, 40)

    # Initialize the Architecture
    unet_standard = Architecture(depth=4, activation="softmax",
                                 batch_normalization=True)

    # Create the Neural Network model
    model = Neural_Network(preprocessor=pp, architecture=unet_standard,
                           loss=tversky_crossentropy,
                           metrics=[tversky_loss, dice_soft, dice_crossentropy],
                           batch_queue_size=3, workers=3, learninig_rate=0.001)

    # Dump model to disk for reproducibility
    model.load(model_file)
    return model, data_io


initalize()
model, data_io = get_model()


def run_segmentation_task(patient_id, result_id):
    my_logger.info("patient_id= {} will predict with this result_id={}".format(patient_id, result_id))
    current_date = str(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))

    try:
        # turn stage to RUNNING
        new_process = mongo.covid19.results.insert({
            "patient_id": str(patient_id),
            "result_id": result_id,
            "status": "READY",
            "date": current_date[:10],
            "time": current_date[11:]

            })

        my_logger.info(" Task result_id={} patient_id= {} in READY STAGE ".format(result_id,patient_id))

        # dicom process
        mongo.covid19.results.update({"patient_id": str(patient_id),"result_id":result_id},{"$set":{"status":"PROCESSING"}})
        my_logger.info(" Task result_id={} patient_id= {} in PROCESSING STAGE ".format(result_id,patient_id))
        # Check is exist patient dicoms

        try:
            patient_info = dcm2nifti(patient_id)
        except Exception:
            my_logger.error(" Task result_id={} patient_id= {} dicoms couldnt convert to nifti ".format(result_id,patient_id))
            return True

        mongo.covid19.results.update(
            {"patient_id": str(patient_id), "result_id": result_id}, {"$set": patient_info})


        my_logger.info("Prediction is starting for {}...".format(str(patient_id)))

        predictions = model.predict([str(patient_id)], return_output=True)
        results = export_results(predictions, data_io, str(patient_id),patient_info)

        del predictions
        mongo.covid19.results.update({"patient_id":str(patient_id),"result_id":result_id},{"$set":results})
        my_logger.info(" Task result_id={} patient_id= {} in DONE STAGE ".format(result_id,patient_id))
        # CLEAN PATIENT FOLDERS
        clean_patient(str(patient_id))
        my_logger.info(" Task result_id={} patient_id= {} folders CLEANING ".format(result_id,patient_id))
        del results

    except TypeError as e:
        my_logger.error("Type Error result_id={} patient_id= {}  ".format(result_id,patient_id), exc_info=True)
        mongo.covid19.results.update({"patient_id": str(
            patient_id), "result_id": result_id}, {"$set": {"status": "ERROR"}})

    except FileNotFoundError as e:
        my_logger.error("FileNotFoundError result_id={} patient_id= {}  ".format(result_id,patient_id), exc_info=True)
        mongo.covid19.results.update({"patient_id": str(
            patient_id), "result_id": result_id}, {"$set": {"status": "ERROR"}})

    except FolderStructureError as e:
        mongo.covid19.results.update({"patient_id": str(
            patient_id), "result_id": result_id}, {"$set": {"status": "INVALID"}})
        my_logger.error(e.getMessage())

    except Exception as e:
        my_logger.error("Unexcepted result_id={} patient_id= {} ".format(result_id,patient_id), exc_info=True)
        mongo.covid19.results.update({"patient_id": str(
            patient_id), "result_id": result_id}, {"$set": {"status": "ERROR"}})
