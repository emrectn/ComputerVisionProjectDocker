import os
from pymongo import MongoClient

mongo = MongoClient(os.environ["MONGO_URI"])