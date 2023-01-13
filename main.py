# Include libraries
from pade.misc.utility import display_message, start_loop, call_later
from pade.core.agent import Agent
from pade.acl.messages import ACLMessage
from pade.acl.aid import AID
from sys import argv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from collections import Counter
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv1D, MaxPooling1D, Dropout, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import seaborn as sn
import time
from sklearn.preprocessing import QuantileTransformer


if __name__ == '__main__':

    agents = list()

    preprocessingagent = Preprocessing_agent(AID(name='preprocessing__agent'))
    agents.append(preprocessingagent)

    trainingagent = Training_agent(AID(name='training__agent'))
    agents.append(trainingagent)

    predictionagent = Prediction_agent(AID(name='prediction__agent'))
    agents.append(predictionagent)

    start_loop(agents)