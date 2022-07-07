from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Bidirectional,Permute,Reshape,LSTM, TimeDistributed
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from Helper import log
import numpy as np

def build_model(states,actions):
    # print(f"model: {states}")
    log.info('Building Custom model')
    model = Sequential()
    model.add(Conv2D(12,(3,3), activation='relu',input_shape=states,data_format="channels_last"))
    model.add(Conv2D(18,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(24,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(lr = 0.001),metrics=['accuracy'])
    return model
    
def LSTM_model(states,actions):
    # print(f"model: {states}")
    log.info('Building Custom LSTM model')
    model = Sequential()
    model.add(Conv2D(12,(3,3), activation='relu',input_shape=states,data_format="channels_last"))
    model.add(Conv2D(18,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(24,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(3,3), activation='relu'))
    #reshape from 4d to 3d data
    # model.add(Permute((2,1,3)))
    # model.add(Flatten())
    model.add(Reshape((32,812)))
    model.add(Bidirectional(LSTM(20,return_sequences=True)))
    model.add(Bidirectional(LSTM(20,return_sequences=True)))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(lr = 0.001),metrics=['accuracy'])
    return model

def transfer_model(states,action):
    log.info('Building with transfer learned xception model')
    base_model = Xception(weights="imagenet", include_top=False, input_shape=states)
    base_model.trainable = False ## Not trainable weights
    flatten_layer = Flatten()
    dense_layer_1 = Dense(500, activation='relu')
    dense_layer_2 = Dense(300, activation='relu')
    dense_layer_3 = Dense(100, activation='relu')
    prediction_layer = Dense(action, activation='linear')


    model = Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        dense_layer_3,
        prediction_layer
    ])
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(lr = 0.001),metrics=['accuracy'])
    return model

class CustomProcessor(Processor):
    '''
    acts as a coupling mechanism between the agent and the environment
    '''
    def process_state_batch(self, batch):
        batch = np.array(batch, dtype=object)
        batch = np.squeeze(batch,axis=1)
        return batch

processor = CustomProcessor()

def build_agent(model, actions):
    log.info('Building agent')
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,processor=processor,
                  nb_actions=actions, nb_steps_warmup=500, target_model_update=1e-3)
    return dqn

    