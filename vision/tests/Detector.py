import os
from PIL import Image
from timeit import default_timer as timer

from utils.utils import load_extractor_model, load_features, detect_object
from utils.dicom_to_jpg import FolderStructureError
# import test
import utils
import pandas as pd
import numpy as np
from utils.Get_File_Paths import GetFileList
from models.src.keras_yolo3.yolo import YOLO, detect_video
import random

import time
import re
#import h2o
from scipy.signal import convolve2d as conv2
from datetime import datetime

from logger import MyLogger
from logging import DEBUG, INFO, ERROR, CRITICAL

my_logger = MyLogger("detector",level=DEBUG)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up folder names for default values
model_weights = os.path.join('weights', 'trained_weights_final.h5')
model_classes = os.path.join('weights', 'data_classes.txt')
anchors_path = os.path.join('models', 'src', 'keras_yolo3', 'model_data', 'yolo_anchors.txt')

def get_img_size(img_path):
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    y_size, x_size, _= np.array(image).shape
    return y_size, x_size

def covid_detector(yolo, FLAGS):
    labels = get_labels()
    current_date = str(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    # initialize patient result
    data = dict()
    data["patient_id"] = str(FLAGS["patient_id"])
    data["result_id"] = FLAGS["result_id"]
    data["results"] = {}
    data["symptom"] ={}
    data["status"] = "DONE"
    data["isCovid"] = None
    data["date"] = current_date[:10]
    data["time"] = current_date[11:]

    #data["predicted_class"] =  int(predicted_class)
    #data["predicted_confidence"] = float(predicted_confidence)
    data["predicted_class"] = 0
    data["predicted_confidence"] = 0

    #if you want to specify
    input_paths = GetFileList(FLAGS['input_path'], endings=['.jpg'])
    # Split images and videos
    img_endings = ('.jpg', '.jpg', '.png')
    input_image_paths = []

    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)


    if input_image_paths:
        my_logger.info('Found {} input images: {} ...'.format(len(input_image_paths), [os.path.basename(f) for f in input_image_paths[:5]]))
        if len(input_image_paths) < 200:
            raise(FolderStructureError("Number of dcm is less than 200"))
        start = timer()

        # initialize image size, The problem occurs because the first pictures are passed.
        y_size, x_size = get_img_size(input_image_paths[0])

        # Get ignore percentage from enviroment
        instance_min = int(len(input_image_paths) * float(os.environ["IGNORE_FIRST"]))
        instace_max = int(len(input_image_paths) * (1-float(os.environ["IGNORE_LAST"])))

        for i, img_path in enumerate(input_image_paths):

            result = dict()
            box = dict()

            instanceNum = int(re.split("[_]{2,}", img_path)[1].split(".")[0])

            #PREDICT IMAGE
            if instanceNum > instance_min and instanceNum < instace_max:
                prediction, image = detect_object(yolo, img_path, save_img=False, save_img_path=FLAGS['output'],postfix='')
                y_size, x_size,_ = np.array(image).shape
            else:
                prediction = []

            data["results"][str(instanceNum)] = result
            result["instanceNumber"] = instanceNum
            result["path"] = img_path.rstrip('\n')
            result["boxes"] = []

            #FOUNDED BOX(ES)
            for single_prediction in prediction:

                # calc area of the detection
                xmin = int(single_prediction[0])
                ymin = int(single_prediction[1])
                xmax = int(single_prediction[2])
                ymax = int(single_prediction[3])

                area = abs(xmax - xmin + 1) * (ymax - ymin + 1)

                if area > int(os.environ["AREA_THRESHOLD"]):
                    result["isCovid"] = 1
                    label = labels[int(single_prediction[4])]
                    box = {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "label": label,
                        "confidence": float(single_prediction[5]),
                        "x_size": x_size,
                        "y_size": y_size }

                    if label in data["symptom"]:
                        data["symptom"][label] += 1
                    else:
                        data["symptom"][label] = 1
                    data["results"][str(instanceNum)]["boxes"].append(box)

        result_sorted = sorted(data["results"].items(), key=lambda x: int(x[0]))
        data["results"] = [values for key, values in result_sorted]

        end = timer()
        my_logger.info('Processed {} images in {:.1f}sec - {:.1f}FPS'.format(len(input_image_paths), end-start, len(input_image_paths)/(end-start)))

    # Close the current yolo session
    #yolo.close_session()
    return data



#read sample input, all functions will write to df
#df = pd.read_csv('df_result_1_haziran_normal.csv')
H2O_MODEL_PATH = "models/11_haziran_h2o"


# def get_labels():
#     LABELS = os.getenv("LABELS")
#     return LABELS.split(",")


def get_labels():
    with open("weights/data_classes.txt", "r") as f:
        labels = f.readlines()
    # Remove newline char, Buzlucam\n
    return list(map(lambda x: x.strip(), labels))

###################
# Predict Covid PreProcess
##################
def predict_format(csv_file):

    df = pd.read_csv(csv_file)
    df[["instance","patient"]] = df["image"].apply(get_patient)
    df = df[["patient","instance","confidence"]]
    df["instance"] = df["instance"].astype(str)
    df["instance"] = df["instance"].apply(lambda x: x.zfill(3))
    df = df.pivot_table(index=['patient'], columns='instance',
                     values='confidence', aggfunc='sum').reset_index()


    df= df.reindex(sorted(df.columns),axis=1)

    return df

def get_patient(row):
    row_split = row.split("_")
    patient = "".join(row_split[:-2])

    return pd.Series(dict({"patient": patient, "instance":row_split[-1][:-4]}))

###################
# Predict Covid
##################
def create_seq(df):
    #create sequence for dicoms
    n = 20
    v = np.vstack([(conv2(df.values!=0,[[1]*I])==I).sum(1) for I in range(2,n+1)]).T
    df_v = pd.DataFrame(v, columns = [[str(i)+'x' for i in range(2,n+1)]])
    df = pd.concat([df, df_v], 1)
    return df
def gen_feature(df, col_count):
    #generate new features for model input
    df['max_value'] = df.max(axis=1)
    df['sum_value'] = df.iloc[:, 0:col_count].sum(axis=1)
    df['zero_count'] = (df == 0).sum(axis=1)
    df['nonzero_count'] = col_count - df['zero_count']
    df['avg'] = df['sum_value'] / df['nonzero_count']
    df['covid_count_ratio'] = df['nonzero_count'] / col_count
    df.drop(columns=['zero_count', 'nonzero_count'], inplace=True)
    return df
def merge_dfs(df_seq, df):
    # merge the sequences and features
    cols = ['patient', 'max_value', 'sum_value', 'avg', 'covid_count_ratio']
    df_seq = df_seq.merge(df[cols], on=['patient'])
    return df_seq
def manipulate_df(df,col_count):
    # delete dicom probabilities and 2x sequences
    del_dicom_cols = df.columns[0:col_count]
    df.drop(del_dicom_cols, axis=True, inplace = True)
    df.drop([('2x',)], axis=True, inplace = True)
    return df

def run(csv):
# df is the output of the yolo model with ratios
# df = assign the df first

    df = csv.fillna(0)
    col_count =  df.shape[1] - 1
    df_seq = create_seq(df)
    df = gen_feature(df, col_count)
    df = merge_dfs(df_seq, df)
    df = manipulate_df(df,col_count)


    h2o.init(max_mem_size='50M', nthreads=1)

    #change model path to the model which is under the models
    #modelname is DeepLearning_grid__1_AutoML_20200608_183808_model_7
    #model_path = 'C:\\Users\\IS97897\\Desktop\\h20_documents\\DeepLearning_grid__1_AutoML_20200608_183808_model_7'

    #load the model
    model = h2o.load_model(H2O_MODEL_PATH)
    #change df to h2o frame
    #df is coming from covid_model_prediction
    hf = h2o.H2OFrame(df)
    #predict the results
    prediction = model.predict(hf).as_data_frame()
    print(prediction)
    print(hf)
    h2o.shutdown()

    # df will be the input of the ML model
    # after predict function you will get the results of probability of being Covid
    return prediction["predict"].iloc[0], prediction["p1"].iloc[0]




