import os
import time
from db import mongo
from vision import run_segmentation_task

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from logger import MyLogger
from logging import DEBUG

my_logger = MyLogger("watcher", level=DEBUG)
my_logger.info("Watcher is started now")

folder_last_modifiy_time = dict()
PATH = os.environ["DICOM_ROOT"]
os.makedirs(PATH, exist_ok=True)


def on_started():
    list_dir = os.listdir(PATH)
    results = mongo.covid19.results.find({})
    # patient_list = [x["patient_id"] for x in results]
    patient_list = [x["patient_id"] if "invalid_pid" not in x else x['invalid_pid'] for x in results]

    my_logger.info("Senkronizasyon başlatılıyor.")
    for i in list_dir:
        if i not in patient_list and i != ".DS_Store":
            my_logger.info("İşlenmemiş dosya bulundu: {} . İşleme alınıyor.".format(i))
            new_result_id = int(time.time()*1000)

            run_segmentation_task(i, new_result_id)
    my_logger.info("Senkronizasyon bitti.")


def update_list(folder):
    # folder = folder.replace("\\", "/").rsplit("/", 1)[0].replace("/IMAGES", "")
    folder = folder.replace("\\", "/").replace("/IMAGES", "")
    folder_last_modifiy_time[folder] = time.perf_counter()


def check_time_out():
    #my_logger.info("CHECK TIME: " + str(folder_last_modifiy_time))
    if not folder_last_modifiy_time:
        return None

    for i in folder_last_modifiy_time.copy():
        timer = time.perf_counter() - folder_last_modifiy_time[i]

        if timer > 10:
            del folder_last_modifiy_time[i]
            if PATH.replace("/", "") != i.replace("/", "") and i.split("/")[-1] != ".DS_Store":
                my_logger.info("Aktarım tamamlandı: {}".format(i))
                new_result_id = int(time.time()*1000)
                patient_id = i.split("/")[-1]
                check = mongo.covid19.results.find({"patinent_id": patient_id})
                if check.count() > 0:
                    my_logger.warn("{} BU HASTA DATABASEDE MEVCUT: PREDICT EDILMEYECEK ".format(patient_id))
                else:
                    run_segmentation_task(patient_id, new_result_id)


def on_created(event):
    if os.path.isdir(event.src_path):
        my_logger.info("New folder created {}".format(event.src_path))
    update_list(event.src_path)


def on_modified(event):
    # my_logger.info("File is modified {}".format(event.src_path))
    pass


def on_moved(event):
    # my_logger.info("File moved from {} to {}'.".format(event.src_path, event.dest_path))
    pass


def on_deleted(evet):
    pass


if __name__ == "__main__":
    patterns = "*"
    ignore_patterns = ""
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive)

    my_event_handler.on_created = on_created
    my_event_handler.on_deleted = on_deleted
    my_event_handler.on_modified = on_modified
    my_event_handler.on_moved = on_moved

    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, PATH, recursive=go_recursively)

    my_observer.start()
    try:
        on_started()
        while True:
            check_time_out()
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        my_logger.info("Watcher is stopped now")
        my_observer.join()
    except Exception as e:
        my_logger.error("Bir hata oluştu")
        print(e)
