import pandas as pd
import pyshark
from pade.core.agent import Agent
from pade.acl.aid import AID
from pade.core.new_ams import AMS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM
from keras.models import Model
import sqlite3
from datetime import datetime

class DataCollectionAgent(Agent):
    def __init__(self, data_collection_strategy, interface=None, pcap_file=None):
        self.aid = AID(name='data_collection_agent')
        super().__init__(aid=self.aid)
        self.data_collection_strategy = data_collection_strategy
        self.interface = interface
        self.pcap_file = pcap_file
        self.ams = AMS()
        self.latency_log = []

    def on_start(self):
        self.receive_request()

    def receive_request(self):
        if self.data_collection_strategy == 'online':
            self.capture_data()
        else:
            self.load_data()

    def capture_data(self):
        """
        Code for online data collection
        """
        # Create a capture object
        try:
            capture = pyshark.LiveCapture(interface=self.interface)
        except Exception as e:
            print("Error: ", str(e))
            return
        # Start the capture
        capture.sniff(packet_count=100)

        # Store the captured data
        self.data = pd.DataFrame(pkt.__dict__ for pkt in capture)

        # Send the data to the next agent
        self.send_data()

    def load_data(self):
        """
        Code for offline data collection
        """
        try:
            capture = pyshark.FileCapture(self.pcap_file)
        except Exception as e:
            print("Error: ", str(e))
            return

        # Store the captured data
        self.data = pd.DataFrame(pkt.as_dict() for pkt in capture)

        # Send the data to the next agent
        self.send_data()

    def send_data(self):
        start_time = time.time()
        self.ams.send_msg(to='preprocessing_agent', sender=self.name, content=self.data)
        end_time = time.time()
        self.latency_log.append(('data_collection_agent', 'preprocessing_agent', end_time - start_time))

class PreprocessingAgent(Agent):
    def __init__(self):
        super().__init__(aid='preprocessing_agent')
        self.latency_log = []

    def on_start(self):
        self.receive_data()

    def receive_data(self):
        start_time = time.time()
        # Preprocess the data
        self.preprocess_data()
        end_time = time.time()
        self.latency_log.append(('preprocessing_agent', 'feature_extraction_agent', end_time - start_time))

    def preprocess_data(self):
        """
        Code for preprocessing the data
        """
        # Convert the data to a pandas dataframe
        self.data = pd.DataFrame(self.data.__dict__['_packets'])

        # Select the relevant columns
        self.data = self.data[['no.', 'time', 'source', 'destination', 'protocol', 'length']]

        # Drop missing values
        self.data.dropna(inplace=True)

        # Send the preprocessed data to the next agent
        self.send_data()
        
    def send_data(self):
        start_time = time.time()
        self.ams.send_msg(to='feature_extraction_agent', sender=self.name, content=self.data)
        end_time = time.time()
        self.latency_log.append(('preprocessing_agent', 'feature_extraction_agent', end_time - start_time))

"""
      def receive_data(self):
        # Extract features from the data
        self.extract_features()
        end_time = time.time()
        self.latency_log.append(('feature_extraction_agent', 'model_training_agent', end_time - start_time))

"""
"""
    def extract_features(self):
        
        #Code for extracting features from the data
        
        # Encode the categorical variables
        self.data['source'] = self.label_encoder.fit_transform(self.data['source'])
        self.data['destination'] = self.label_encoder.fit_transform(self.data['destination'])
        self.data['protocol'] = self.label_encoder.fit_transform(self.data['protocol'])

        # Send the extracted features to the next agent
        self.send_data()

    def send_data(self):
        start_time = time.time()
        self.ams.send_msg(to='feature_extraction_agent', sender=self.name, content=self.data)
        end_time = time.time()
        self.latency_log.append(('feature_extraction_agent', 'model_training_agent', end_time - start_time))
"""

class FeatureExtractionAgent(Agent):
   def init(self):
       super().init(aid='feature_extraction_agent')
       self.label_encoder = LabelEncoder()

   def on_start(self):
       self.receive_data()

   def receive_data(self):
       start_time = time.time()
       # Extract features from the data
       self.extract_features
       end_time = time.time()       
       self.latency_log.append(('feature_extraction_agent', 'model_training_agent', end_time - start_time))


   def extract_features(self):
       """
       Code for extracting features from the data
       """
       # Encode the categorical variables
       self.data['source'] = self.label_encoder.fit_transform(self.data['source'])
       self.data['destination'] = self.label_encoder.fit_transform(self.data['destination'])
       self.data['protocol'] = self.label_encoder.fit_transform(self.data['protocol'])

       # Send the extracted features to the next agent
       self.send_data()

   def send_data(self):
       start_time = time.time()
       self.ams.send_msg(to='model_training_agent', sender=self.name, content=self.data)
       end_time = time.time()
       self.latency_log.append(('feature_extraction_agent', 'model_training_agent', end_time - start_time))
 
class ModelTrainingAgent(Agent):
    def __init__(self):
        super().__init__(aid='model_training_agent')
        self.latency_log = []
        self.models = {'RandomForestClassifier': RandomForestClassifier(),
                       'ANN': Sequential(),
                       'CNN': Sequential(),
                       'LSTM': Sequential()}
        self.models_history = {'RandomForestClassifier': {'loss': [], 'acc': [], 'time': [], 'power_consumption': [], 'mse': []},
                               'ANN': {'loss': [], 'acc': [], 'time': [], 'power_consumption': [], 'mse': []},
                               'CNN': {'loss': [], 'acc': [], 'time': [], 'power_consumption': [], 'mse': []},
                               'LSTM': {'loss': [], 'acc': [], 'time': [], 'power_consumption': [], 'mse': []}}
        self.create_ann_model()
        self.create_cnn_model()
        self.create_lstm_model()

    def create_ann_model(self):
        """
        Code for creating and compiling the ANN model
        """
        self.models['ANN'].add(Dense(units=64, activation='relu', input_shape=(self.features.shape[1],)))
        self.models['ANN'].add(Dense(units=32, activation='relu'))
        self.models['ANN'].add(Dense(units=1, activation='sigmoid'))
        self.models['ANN'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def create_cnn_model(self):
        """
        Code for creating and compiling the CNN model
        """
        self.models['CNN'].add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.features.shape[1], 1)))
        self.models['CNN'].add(MaxPooling1D(pool_size=2))
        self.models['CNN'].add(Flatten())
        self.models['CNN'].add(Dense(50, activation='relu'))
        self.models['CNN'].add(Dense(1, activation='sigmoid'))
        self.models['CNN'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def create_lstm_model(self):
        """
        Code for creating and compiling the LSTM model
        """
        self.models['LSTM'].add(LSTM(units=50, input_shape=(self.features.shape[1], 1)))
        self.models['LSTM'].add(Dense(1, activation='sigmoid'))
        self
        
    def on_start(self):
        self.receive_data()

    def receive_data(self):
        start_time = time.time()
        # Train the models
        self.train_models()
        end_time = time.time()
        self.latency_log.append(('model_training_agent', 'model_predictor_agent', end_time - start_time))

    def train_models(self):
        """
        Code for training the models
        """
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2)
        for model_name in self.models:
            start_time = time.time()
            history = self.models[model_name].fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            self.models_history[model_name]['loss'].append(history.history['loss'])
            self.models_history[model_name]['acc'].append(history.history['accuracy'])
            self.models_history[model_name]['time'].append(end_time - start_time)
            self.models_history[model_name]['power_consumption'].append(compute_power_consumption(end_time - start_time))
            y_pred = self.models[model_name].predict(X_test)
            self.models_history[model_name]['mse'].append(mean_squared_error(y_test, y_pred))

        # Send the trained models to the next agent
        self.send_models()

    def send_models(self):
        start_time = time.time()
        self.ams.send_msg(to='model_predictor_agent', sender=self.name, content=self.models)
        end_time = time.time()
        self.latency_log.append(('model_training_agent', 'model_predictor_agent', end_time - start_time))

class ModelPredictorAgent(Agent):
    def __init__(self):
        super().__init__(aid='model_predictor_agent')
        self.latency_log = []

    def on_start(self):
        self.receive_models()

    def receive_models(self):
        start_time = time.time()
        self.models = self.recv_msg()
        end_time = time.time()
        self.latency_log.append(('model_training_agent', 'model_predictor_agent', end_time - start_time))

        # Evaluate the models
        self.evaluate_models()

    def evaluate_models(self):
        """
        Code for evaluating the models
        """
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2)
        for model_name in self.models:
            y_pred = self.models[model_name].predict(X_test)
            loss, acc = self.models[model_name].evaluate(X_test, y_test, verbose=0)
            self.models_history[model_name]['loss'].append(loss)
            self.models_history[model_name]['acc'].append(acc)
            self.models_history[model_name]['mse'].append(mean_squared_error(y_test, y_pred))

        # Send the models' performance to the next agent
        self.send_models_performance()

    def send_models_performance(self):
        start_time = time.time()
        self.ams.send_msg(to='quality_of_service_agent', sender=self.name, content=self.models_history)
        end_time = time.time()
        self.latency_log.append(('model_predictor_agent', 'quality_of_service_agent', end_time - start_time))
        
        
class QualityOfServiceAgent(Agent):
    def __init__(self):
        super().__init__(AID(name='quality_of_service_agent'))
        self.lstm_model = None
        self.received_data = None
        self.predicted_throughput = None
        self.communication_latency = None
        self.latency_log = []

    def on_start(self):
        self.receive_data()
        self.predict_throughput()
        self.plot_results()
        self.log_performance()
        
        
    def receive_models(self):
        start_time = time.time()
        self.models = self.recv_msg()
        end_time = time.time()
        self.latency_log.append(('quality_of_service_agent', 'model_predictor_agent', end_time - start_time))

    def receive_data(self):
        self.received_data = self.recv(timeout=10)
        self.communication_latency = datetime.now() - self.received_data.metadata['timestamp']

    def predict_throughput(self):
        link_data = self.received_data.content
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(50, input_shape=(link_data.shape[1], 1)))
        self.lstm_model.add(Dense(1))
        self.lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        self.lstm_model.fit(link_data, self.predicted_throughput, epochs=100, batch_size=32)
        self.predicted_throughput = self.lstm_model.predict(link_data)

    def plot_results(self):
        plt.plot(self.predicted_throughput)
        plt.title('Predicted throughput for the next 6 hours')
        plt.xlabel('Time (hours)')
        plt.ylabel('Throughput (Mbps)')
        plt.show()

    def log_performance(self):
        print(f'Communication latency: {self.communication_latency}')
        
    def log_latency(self):
        # Print the communication latency between each agent
        for i, latency in enumerate(self.latency_log):
            print(f'Latency between agent {i} and {i+1}: {latency}')


if __name__ == '__main__':
    # Define your data collection strategy here
    data_collection_strategy = "offline"
    if data_collection_strategy == "online":
        interface = " Wi-Fi"
        data_collection_agent = DataCollectionAgent(data_collection_strategy=data_collection_strategy, interface=interface)
    elif data_collection_strategy == "offline":
        pcap_file = "C:/Users/pmushidi.ECE-D71895/Desktop/MAS January 19 2023/Dataset30M.pcap"
        data_collection_agent = DataCollectionAgent(data_collection_strategy=data_collection_strategy, pcap_file=pcap_file)
    else:
        print("Invalid data collection strategy")
        exit()

    preprocessing_agent = PreprocessingAgent()
    feature_extraction_agent = FeatureExtractionAgent()
    model_training_agent = ModelTrainingAgent()
    model_predictor_agent = ModelPredictorAgent()
    quality_of_service_agent = QualityOfServiceAgent()

    data_collection_agent.start()
    preprocessing_agent.start()
    feature_extraction_agent.start()
    model_training_agent.start()
    model_predictor_agent.start()
    quality_of_service_agent.start()






