# AI Final Project
# Group 2
# Parker Bray, Robert Kellems, Cole Metzger, Kai Sandstrom
# Sound Classification

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import librosa
import matplotlib.pyplot as plt
from librosa import display
import os.path

drivepath = "/content/drive/My Drive/AI Final Project/"
localpath = "./"
filemode =  "null"

def process_sounds(raw, mean):
  """
  Creates CSV file(s) depending on the values of the booleans raw and mean.
  If raw is True, CSV containing spectrogram data is created
  If mean is True, CSV containing standardized spectrograms used as network input is created
  """
  if filemode == "drive":
    basepath = drivepath
  else:
    basepath = localpath
  path = basepath + "UrbanSound8K/UrbanSound8K"
  oldcsv = path + "/metadata/UrbanSound8K.csv"
  rawcsv = basepath + "DataWithSpectros.csv"
  meancsv = basepath + "DataWithSpectrosInput.csv"
  spectros = []
  means = []
  if (raw and not mean):
    data = pd.read_csv(rawcsv)
  else:
    data = pd.read_csv(oldcsv)
  datalen = len(data)
  for i in range(datalen):
    if (raw and not mean):
      spectro = eval(data.iloc[i][0])
      spectro = np.array(spectro)
      mean = np.mean(spectro.T, axis=0)
      means.append(mean.tolist())
    else:
      filepath = path + "/audio/fold" + str(data.iloc[i][5]) + "/" + data.iloc[i][0]
      y, sr = librosa.load(filepath)
      nplist = librosa.feature.melspectrogram(y, sr)
      spectros.append(nplist.tolist())
      if (not mean):
        nplist = np.mean(nplist.T, axis=0)
        means.append(nplist.tolist())
    if i%1000 == 0:
      print(f"processed {i}/{datalen}")
  if not raw:
    data['Spectrograms'] = spectros
    data.to_csv(rawcsv)
  if not mean:
    data['Spectrograms'] = means
    data.to_csv(meancsv)

def display_spectrogram(data, loc):
  """
  Given a DataFrame containing each of the spectrograms in the dataset (data) and the row number of the 
  desired spectrogram (loc), displays a visual representation of the spectrogram.
  """
  spectro = data.at[loc, 'Spectrograms']
  spectro = eval(spectro) # converting string representation of spectrogram into usable array 
  spectro = np.array(spectro) # ^
  # code for creating plot from: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
  fig, ax = plt.subplots()
  spectro_dB = librosa.power_to_db(spectro, ref=np.max)
  img = librosa.display.specshow(spectro_dB, x_axis='time', y_axis='mel', sr=22050, fmax=8000, ax=ax)
  fig.colorbar(img, ax=ax, format='%+2.0f dB')
  ax.set(title='Mel-frequency spectrogram')
  plt.show()

def create_network_input(data, classification, testFold, sound):
  """
  Given a DataFrame containing each of the spectrograms in the dataset (data), a string representing the 
  desired classification type (classification), the number of the fold designated for testing (testFold), 
  and, in the case of binary classification, the number assigned to the sound class which the network 
  will be attempting to separate from the rest of the sounds (sound), creates and returns 4 arrays;
  trainFeatures and testFeatures contain the spectrograms in the training and testing sets respectively 
  in a standardized form in order to be used as network input, while trainLabels and testLabels contain
  the classes for each of the sounds in the training and testing set, respectively.
  """
  features = data['Spectrograms'].to_numpy()
  labels = data['classID'].to_numpy()
  trainFeatures, testFeatures, trainLabels, testLabels = [],[],[],[]
  for i in range(len(data)):
    if classification == 'binary':
      if labels[i] == sound:
        label = 1
      else:
        label = 0
    else:
      label = labels[i]
    if data.at[i, 'fold'] == testFold:
      feature = eval(features[i]) # this is done to convert string representation from data to actual usable array
      feature = np.array(feature) # ^
      testFeatures.append(feature)
      testLabels.append(label)
    else:
      feature = eval(features[i]) # same as above
      feature = np.array(feature)
      trainFeatures.append(feature) 
      trainLabels.append(label)
  trainFeatures = np.array(trainFeatures)
  testFeatures = np.array(testFeatures)
  trainLabels = pd.get_dummies(trainLabels).to_numpy()
  testLabels = pd.get_dummies(testLabels).to_numpy()
  return trainFeatures, testFeatures, trainLabels, testLabels

def binary_classify(data, soundclass):
  """
  Given a DataFrame containing each of the spectrograms in the dataset (data) and a number representing
  the desired sound class to separate from the rest of the data (soundclass), creates a simple neural
  network to classify each sound as either belonging to soundclass or not.
  Lines which are commented out reflect previous version of the function in which only accuracy and
  loss values were kept track of.
  """
  #accuracies, losses = [], []
  accumulator = []
  for j in range(10): # iterating through each sound class
    #accuracy, loss = [], []
    if soundclass != -1:
      j = soundclass
    confusions = []
    for i in range(1,11): # iterating through each folder; i is used as the testing folder
      print(f"Sound {j}")
      print(f"Test fold {i}:")
      # creating necessary data for network
      trainFeatures, testFeatures, trainLabels, testLabels = create_network_input(data, "binary", i, j)

      # creating a network
      network = Sequential()
      network.add(Dense(128, input_shape=(128,), activation = 'relu'))
      network.add(Dense(256, activation = 'relu'))
      network.add(Dense(2, activation = 'softmax'))
      network.compile(loss='binary_crossentropy', metrics=['accuracy'])
      history = network.fit(trainFeatures, trainLabels, batch_size=64, epochs=30)
      print(history.history)
      
      # in original verison of this function, network is tested here
      #results = network.evaluate(testFeatures, testLabels, batch_size=64)
      #print(results)
      #loss.append(results[0])
      #accuracy.append(results[1])

      # network's predictions are compared to the labels, resulting testing data is collected
      raw_predictions = network.predict(testFeatures, batch_size=64)
      predictions = []
      for i in range(len(raw_predictions)):
        if raw_predictions[i][0] > raw_predictions[i][1]:
          predictions.append(0)
        else:
          predictions.append(1)
      testLabels = [i[1] for i in testLabels]
      confusionTensor = tf.math.confusion_matrix(testLabels, predictions)
      confusion = confusionTensor.numpy()
      acc = (confusion[0][0]+confusion[1][1]) / (confusion[0][0]+confusion[0][1]+confusion[1][0]+confusion[1][1])
      accs = [acc, confusion[1][1], confusion[0][0], confusion[0][1], confusion[1][0]]
      print(accs)
      confusions.append(accs)

    # in original function, average accuracy/loss values for given sound class being separated are collected
    #print(accuracy)
    #print(loss)
    #avgacc = sum(accuracy)/10
    #avglos = sum(loss)/10
    #print("Average loss: " + str(avglos))
    #accuracies.append(avgacc)
    #losses.append(avglos)

    # average accuracy and confusion matrix values are collected
    avgacc = sum([i[0] for i in confusions])/10
    tp = sum([i[1] for i in confusions]) # true positive
    tn = sum([i[2] for i in confusions]) # true negative
    fp = sum([i[3] for i in confusions]) # false positive
    fn = sum([i[4] for i in confusions]) # false negative
    print("Average accuracy: " + str(avgacc))
    print("Total confusion values (tp, tn, fp, fn):")
    print(f"{tp}, {tn}, {fp}, {fn}")
    if soundclass != -1:
      return
    accumulator.append([avgacc, tp, tn, fp, fn])

  # printing final results in original version of function
  # print("Average accuracy, loss for each class:")
  # print(f"air_conditioner:  {accuracies[0]}, {losses[0]}")
  # print(f"car_horn:         {accuracies[1]}, {losses[1]}")
  # print(f"children_playing: {accuracies[2]}, {losses[2]}")
  # print(f"dog_bark:         {accuracies[3]}, {losses[3]}")
  # print(f"drilling:         {accuracies[4]}, {losses[4]}")
  # print(f"engine_idling:    {accuracies[5]}, {losses[5]}")
  # print(f"gun_shot:         {accuracies[6]}, {losses[6]}")
  # print(f"jackhammer:       {accuracies[7]}, {losses[7]}")
  # print(f"siren:            {accuracies[8]}, {losses[8]}")
  # print(f"street_music:     {accuracies[9]}, {losses[9]}")
  # print()
  # print("Global average accuracy: " + str(sum(accuracies)/10))
  # print("Global average loss:     " + str(sum(losses)/10))

  # printing our final results  
  print()
  print("Full binary classifier:")
  print()
  print("For each class: average acc, total tp, total tn, total fp, total fn:")
  print(f"air_conditioner:  {accumulator[0][0]} {accumulator[0][1]} {accumulator[0][2]} {accumulator[0][3]} {accumulator[0][4]}")
  print(f"car_horn:         {accumulator[1][0]} {accumulator[1][1]} {accumulator[1][2]} {accumulator[1][3]} {accumulator[1][4]}")
  print(f"children_playing: {accumulator[2][0]} {accumulator[2][1]} {accumulator[2][2]} {accumulator[2][3]} {accumulator[2][4]}")
  print(f"dog_bark:         {accumulator[3][0]} {accumulator[3][1]} {accumulator[3][2]} {accumulator[3][3]} {accumulator[3][4]}")
  print(f"drilling:         {accumulator[4][0]} {accumulator[4][1]} {accumulator[4][2]} {accumulator[4][3]} {accumulator[4][4]}")
  print(f"engine_idling:    {accumulator[5][0]} {accumulator[5][1]} {accumulator[5][2]} {accumulator[5][3]} {accumulator[5][4]}")
  print(f"gun_shot:         {accumulator[6][0]} {accumulator[6][1]} {accumulator[6][2]} {accumulator[6][3]} {accumulator[6][4]}")
  print(f"jackhammer:       {accumulator[7][0]} {accumulator[7][1]} {accumulator[7][2]} {accumulator[7][3]} {accumulator[7][4]}")
  print(f"siren:            {accumulator[8][0]} {accumulator[8][1]} {accumulator[8][2]} {accumulator[8][3]} {accumulator[8][4]}")
  print(f"street_music:     {accumulator[9][0]} {accumulator[9][1]} {accumulator[9][2]} {accumulator[9][3]} {accumulator[9][4]}")
  print()
  print("Global average accuracy: " + str(sum([i[0] for i in accumulator])/10))
  total_tp = sum([i[1] for i in accumulator])
  total_tn = sum([i[2] for i in accumulator])
  total_fp = sum([i[3] for i in accumulator])
  total_fn = sum([i[4] for i in accumulator])
  print("Global total confusion value counts (tp, tn, fp, fn):")
  print(f"{total_tp} {total_tn} {total_fp} {total_fn}")
  print()

def naive_tenway(data):
  """
  Given a DataFrame containing each of the spectrograms in the dataset, creates a 
  simple neural network to classify each sound as belonging to one of ten different
  sound classes.  
  """
  accuracies, losses = [], []
  for i in range(1,11): # iterating through the folders; i is used as the testing folder
    print("Test fold" + str(i))
    # creating necessary data for network
    trainFeatures, testFeatures, trainLabels, testLabels = create_network_input(data, "tenway", i, 0)

    # creating a network
    network = Sequential()
    network.add(Dense(128, input_shape=(128,), activation = 'relu'))
    network.add(Dense(256, activation = 'relu'))
    network.add(Dense(10, activation = 'softmax'))
    network.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = network.fit(trainFeatures, trainLabels, batch_size=64, epochs=30)
    print(history.history)
      
    # testing our network and collecting the results
    results = network.evaluate(testFeatures, testLabels, batch_size=64)
    print(results)
    losses.append(results[0])
    accuracies.append(results[1])
  # printing final results
  print()
  print("Naive ten-way classifier:")
  print()
  print("Average accuracy: " + str(sum(accuracies)/10))
  print("Average loss:     " + str(sum(losses)/10))
  print()

def improved_tenway(data):
  """
  Given a DataFrame containing each of the spectrograms in the dataset, creates/trains ten 
  binary classifiers like the one created in binary_classify and uses them in tandem to do 
  10-way classification.
  """
  bin_predictions = [[],[],[],[],[],[],[],[],[],[]] 
  for i in range(10):
    for j in range(1,11):
      print(f"Sound {i}")
      print(f"Test fold {j}:")
      # creating necessary data for network
      trainFeatures, testFeatures, trainLabels, testLabels = create_network_input(data, "binary", j, i)

      # creating a network
      network = Sequential()
      network.add(Dense(128, input_shape=(128,), activation = 'relu'))
      network.add(Dense(256, activation = 'relu'))
      network.add(Dense(2, activation = 'softmax'))
      network.compile(loss='binary_crossentropy', metrics=['accuracy'])

      # testing network and collecting results
      history = network.fit(trainFeatures, trainLabels, batch_size=64, epochs=30)
      raw_predictions = network.predict(testFeatures, batch_size=64)
      bin_predictions[i] += raw_predictions.tolist()

  labels = []
  predictions = []
  # create a list of sound class labels ordered by folder
  for i in range(1, 11):
    for j in range(len(data)): 
      if data.iloc[j][6] == i:
        labels.append(data.iloc[j][7])
  # get list of predicted sound classes from list of raw predictions
  for i in range(len(bin_predictions[0])):
    probs = []
    for j in range(10):
      probs.append(bin_predictions[j][i][1])
    predictions.append(probs.index(max(probs)))
  # stores the number of correct/incorrect predictions for each sound class
  counts = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
  # checks whether each prediction matches the sound's actual class
  for i in range(len(predictions)):
    if labels[i] == predictions[i]:
      counts[labels[i]][0] += 1
    else:
      counts[labels[i]][1] += 1
  correct = sum([i[0] for i in counts])
  incorrect = sum([i[1] for i in counts])

  # printing results
  print()
  print("Improved ten-way classifier:")
  print()
  print("Accuracy for each sound class (accuracy, total correct, total incorrect):")
  print(f"air_conditioner:  {counts[0][0]/(counts[0][0]+counts[0][1])} {counts[0][0]} {counts[0][1]}")
  print(f"car_horn:         {counts[1][0]/(counts[1][0]+counts[1][1])} {counts[1][0]} {counts[1][1]}")
  print(f"children_playing: {counts[2][0]/(counts[2][0]+counts[2][1])} {counts[2][0]} {counts[2][1]}")
  print(f"dog_bark:         {counts[3][0]/(counts[3][0]+counts[3][1])} {counts[3][0]} {counts[3][1]}")
  print(f"drilling:         {counts[4][0]/(counts[4][0]+counts[4][1])} {counts[4][0]} {counts[4][1]}")
  print(f"engine_idling:    {counts[5][0]/(counts[5][0]+counts[5][1])} {counts[5][0]} {counts[5][1]}")
  print(f"gun_shot:         {counts[6][0]/(counts[6][0]+counts[6][1])} {counts[6][0]} {counts[6][1]}")
  print(f"jackhammer:       {counts[7][0]/(counts[7][0]+counts[7][1])} {counts[7][0]} {counts[7][1]}")
  print(f"siren:            {counts[8][0]/(counts[8][0]+counts[8][1])} {counts[8][0]} {counts[8][1]}")
  print(f"street_music:     {counts[9][0]/(counts[9][0]+counts[9][1])} {counts[9][0]} {counts[9][1]}")
  print()
  print(f"Total   correct: {correct}")
  print(f"Total incorrect: {incorrect}")
  print(f"Accuracy: {correct/(correct+incorrect)}")
  print()

if __name__ == "__main__":
  while not (filemode == "drive" or filemode == "local"):
    filemode = input("File mode (drive/local): ")
  filemode = 'drive'
  if filemode == "drive":
    from google.colab import drive
    from google.colab import files
    drive.mount('/content/drive/')
    path = drivepath
  else:
    path = localpath
  rawspectros_exists = os.path.isfile(f"{path}DataWithSpectrosInput.csv")
  meanspectros_exists = os.path.isfile(f"{path}DataWithSpectros.csv")
  if not (rawspectros_exists and meanspectros_exists):
    print("Data file(s) missing. Processing dataset (this may take a while)")
    process_sounds(rawspectros_exists, meanspectros_exists)
  data = pd.read_csv(f"{path}DataWithSpectrosInput.csv")

  # code for the user interface
  while(True):
    args = input("Enter a command (--help for list): ").split()
    if len(args) == 1 and args[0] == "--help":
      print("--help")
      print("   Displays a list of valid commands and their syntax")
      print("spectro [fold] [filename]")
      print("   Takes a folder number and filename, displays the file's spectrogram")
      print("   Displaying the spectrogram will halt execution.")
      print("run [test type] [class]")
      print("   Trains a network and runs test depending on mode given.")
      print("   Modes are: binary_full, binary_single, naive_tenway, improved_tenway")
      print("   binary_single takes an extra argument, the sound class number used")
      print("   for binary classification.")
    elif len(args) == 3 and args[0] == "spectro":
      found = -1
      for i in range(len(data)):
        if (str(data.iloc[i][6]) == args[1]) and (data.iloc[i][1] == args[2]):
          found = i
          break
      if found == -1:
        print("File not found.")
      else:
        raw = pd.read_csv(f"{path}DataWithSpectros.csv")
        display_spectrogram(raw, found)
    elif (len(args) == 2 or len(args) == 3) and args[0] == "run":
      if len(args) == 2 and args[1] == 'binary_full':
        binary_classify(data, -1)
      elif len(args) == 2 and args[1] == 'naive_tenway':
        naive_tenway(data)
      elif len(args) == 2 and args[1] == 'improved_tenway':
        improved_tenway(data)
      elif len(args) == 3 and args[1] == 'binary_single':
        try:
          soundclass = int(args[2])
          if soundclass < 0 or soundclass > 9:
            print("Syntax error")
          else:
            binary_classify(data, soundclass)
        except ValueError:
          print("Syntax error")
      else:
        print("Syntax error")
    else:
      print("Invalid syntax. Use --help for a list of commands.")