import numpy as np
import os
import matplotlib.pyplot as plt
import time

from memN2N_data import Data
from MemN2N_model_adj import MemN2N
# from MemN2N_model_rnn import MemN2N

def chooseFiles(file_path, qa_number):
  """
  Choose the training and testing file for one bAbI test.

  :param file_path: Path where the data files for training and testing are stored.
  :param qa_number: Number of the test (from 1 to 20)
  :return: the two names of the files containing training and testing data
  """
  file_list = os.listdir(file_path)
  matching = [s for s in file_list if "qa" + str(qa_number) + "_" in s]

  if "train" in matching[0]:
    train_file = matching[0]
    test_file = matching[1]
  else:
    train_file = matching[1]
    test_file = matching[0]

  return train_file, test_file

  # Choose training and testing files (hard coded)
  # return {
  #   1: ["qa1_single-supporting-fact_train.txt", "qa1_single-supporting-fact_test.txt"],
  #   2: ["qa2_two-supporting-facts_train.txt",   "qa2_two-supporting-facts_test.txt"],
  #   3: ["qa3_three-supporting-facts_train.txt", "qa3_three-supporting-facts_test.txt"],
  #   12: ["qa12_conjunction_train.txt", "qa12_conjunction_test.txt"]
  # }[qa_number]



def parametersValid(params):
  """
  Check if the values in the parameter set 'params' are valid.
  - One value if the same parameter should be used for all runs
  - Multiple values if different parameters should be used for the runs.
    Condition: All parameters with multiple values must have the same number of values!

  :param params: the parameter dictionary
  :return: True if the parameter dictionary is valid, otherwise False
  """
  max_param_len = 0

  for _, element in params.items():
    if max_param_len <= 1 and len(element) > max_param_len:
      max_param_len = len(element)

    if len(element) != 1 and len(element) != max_param_len:
      return False

  return True


def runModel(params, config, data_file_path, delete_summaries, number_embedding_dimensions=20, number_of_hops=3, init_std=0.1):
  """
  Method for SEPARATE training of single tests
  Process the input data, train the MemN2N with the given parameters for the given test and return the test error.

  :param params: parameter dictionary
  :param config: hold the number of the run, is used to get the parameters out of the parameter dictionary
  :param data_file_path: file path, where the training and testing data is stored
  :param delete_summaries: if True, the previous TensorBoard summaries are deleted
  :param number_embedding_dimensions: defines the number of embedding dimensions
  :param number_of_hops: number of hops (layers) of the memory network
  :param init_std: standard deviation of the initialization of the embedding matrices
  :return: the test error rate after training
  """
  test_number           = params["test_number"]           [config if len(params["test_number"]) > 1 else 0]
  batch_size            = params["batch_size"]            [config if len(params["batch_size"]) > 1 else 0]
  memory_size           = params["memory_size"]           [config if len(params["memory_size"]) > 1 else 0]
  number_epochs         = params["number_epochs"]         [config if len(params["number_epochs"]) > 1 else 0]
  initial_learning_rate = params["initial_learning_rate"] [config if len(params["initial_learning_rate"]) > 1 else 0]
  shuffle_training_data = params["shuffle_training_data"] [config if len(params["shuffle_training_data"]) > 1 else 0]
  position_encoding     = params["position_encoding"]     [config if len(params["position_encoding"]) > 1 else 0]
  embedding_lookup      = params["embedding_lookup"]      [config if len(params["embedding_lookup"]) > 1 else 0]
  linear_start          = params["linear_start"]          [config if len(params["linear_start"]) > 1 else 0]

  # INIT data class for data processing
  data = Data(embedding_lookup)
  training_file, testing_file = chooseFiles(data_set, test_number)

  # DATA PROCESSING
  print("\nDATA PROCESSING")
  print("Training file: %s" % training_file)
  train_stories = data.loadData(data_file_path + training_file)
  translated_training_stories, longest_training_story = data.processStories(train_stories)

  print("\nTest file: %s" % testing_file)
  test_stories = data.loadData(data_file_path + testing_file)
  translated_test_stories, longest_test_story = data.processStories(test_stories)

  if memory_size == 0:
    if longest_training_story > longest_test_story:
      memory_size = longest_training_story
    else:
      memory_size = longest_test_story
  print("\nMemory size: %d" % memory_size)

  # TRAINING
  print("\nTRAINING:")
  train_sentences, train_questions, train_answers = data.createMatrices(translated_training_stories,
                                                                        memory_size,
                                                                        batch_size=batch_size,
                                                                        shuffle=shuffle_training_data)
  train_dictionary_size = data.dictionary_size
  data.createInverseDictionary()

  network = MemN2N(dictionary_size=data.dictionary_size,
                   batch_size=batch_size,
                   embedding_dimensions=number_embedding_dimensions,
                   memory_size=memory_size,
                   number_hops=number_of_hops,
                   position_encoding=position_encoding,
                   embedding_lookup=embedding_lookup,
                   linear_start=linear_start,
                   std_deviation=init_std,
                   epochs_lr_annealing=25)

  if delete_summaries:
    network.deleteSummaryFiles()

  network.buildGraph()
  network.train(train_sentences,
               train_questions,
               train_answers,
               initial_learning_rate=initial_learning_rate,
               number_epochs=number_epochs,
               data=data)

  # TESTING
  print("\nTESTING:")
  test_sentences, test_questions, test_answers = data.createMatrices(translated_test_stories,
                                                                     memory_size,
                                                                     batch_size=batch_size,
                                                                     shuffle=False)

  if data.dictionary_size > train_dictionary_size:
    print("ATTENTION: dictionary became bigger!!!")

  # pred_answers, test_error_rate = network.test(test_sentences, test_questions, test_answers, data)
  test_error_rate = network.test(test_sentences, test_questions, test_answers, data)

  network.closeSession()

  return test_error_rate


##############################################################
# MAIN
##############################################################
if __name__ == '__main__':
  # ---------------------------------------------------------
  # Error rates from paper "End-To-End Memory Networks"
  # by Sukhbaatar et al
  # ---------------------------------------------------------
  #                   Test    1     2     3     4     5     6    7     8     9    10    11   12    13   14   15    16    17    18    19    20
  error_rates_paper_BoW   = [0.6, 17.6, 71.0, 32.0, 18.3, 8.7, 23.5, 11.4, 21.1, 22.8, 4.1, 0.3, 10.5, 1.3, 24.3, 52.0, 45.4, 48.1, 89.8, 0.1]
  error_rates_paper_PE    = [0.1, 21.6, 64.2,  3.8, 14.1, 7.9, 21.6, 12.6, 23.3, 17.4, 4.3, 0.3,  9.9, 1.8,  0.0, 52.1, 50.1, 13.6, 87.4, 0.0]
  error_rates_paper_PE_LS = [0.2, 12.8, 58.8, 11.6, 15.7, 8.7, 20.3, 12.7, 17.0, 18.6, 0.0, 0.1,  0.3, 2.0,  0.0,  1.6, 49.0, 10.1, 85.6, 0.0]

  # ---------------------------------------------------------
  # MODEL PARAMETERS
  # ---------------------------------------------------------
  # You can run or multiple tests with the same or different hyperparameters.
  # The numbers of runs are determined by the number of tests specified in "test_number" (see configs_to_run).
  #
  # If you want to use different hyperparameters, you have to specify them as often as you want to run the model.
  # Example (different batch_sizes): "test_number": [1, 6, 6], "batch_size": [32, 32, 16] => working
  # Example (different batch_sizes): "test_number": [1, 6, 6], "batch_size": [32, 16]     => NOT working
  #
  # If you set the memory size to 0, the model will determine the longest story in training and test set,
  # and will adjust the memory size accordingly.
  #
  # Position encoding with "embedding_lookup": [False] is not implemented!

  parameters = {
    "test_number": [2], #range(1,21)
    "batch_size": [32],
    "memory_size": [50],
    "number_epochs": [60],
    "initial_learning_rate":[0.01],
    "shuffle_training_data": [True],
    "embedding_lookup": [False],
    "position_encoding": [False],
    "linear_start": [False]
  }

  # Some tests are very unstable, therefore several random initializations make sense.
  number_initializations = 1

  # More hyperparameters of the network
  number_hops = 3
  embedding_dimensions = 20
  init_std = 0.1

  # For comparing the results with the ones from the paper.
  error_from_paper = error_rates_paper_BoW

  # If True, the program will delete the folders of previous TensorBoard savings
  delete_old_summaries = True
  # Number of runs, which should be run
  configs_to_run = range(len(parameters["test_number"]))

  # Path to the bABI data set
  data_set = "tasks_1-20_v1-2/en/"
  # data_set = "tasks_1-20_v1-2/en-10k/"
  # data_set = "tasks_1-20_v1-2/shuffled/"

  # ---------------------------------------------------------
  # Start the INDEPENDENT training of tests and
  # collect the errors of the testing
  # ---------------------------------------------------------
  start_time_tests = time.time()
  if parametersValid(parameters):
    error_rate_list = []
    duration_list = []

    # Run as many different configurations as there are values in "test_number"
    for run in configs_to_run:
      print("=" * 40)
      print("[%d] Run Test %d:" % (run+1, parameters["test_number"][run]))

      # Run tests multiple times with different random initializations of the embedding matrices
      min_error_rate = 100.
      for trial in range(number_initializations):
        # Delete previous summaries of TensorBoard (but not of the current session)
        del_sum = delete_old_summaries if (run == 0 and trial == 0) else False

        print("-" * 30)
        print("[%d] Run Test %d, Trial %d:" % (run+1, parameters["test_number"][run], trial))
        start_time = time.time()
        error_rate = runModel(parameters, run, data_set, del_sum,
                              number_of_hops=number_hops,
                              number_embedding_dimensions=embedding_dimensions,
                              init_std=init_std)

        # Store the smallest error rate over the multiple random initializations
        if error_rate < min_error_rate:
          min_error_rate = error_rate
          duration = time.time() - start_time

      error_rate_list.append(min_error_rate)
      duration_list.append(duration)

    # Create summary of the tests (error rates, durations and comparision to the error rates from the paper)
    print("\n" + "=" * 40)
    print("Error rates:")
    tolerance_to_paper = 0.02
    for rate, number, duration in zip(error_rate_list, range(len(error_rate_list)), duration_list):
      if rate <= error_from_paper[parameters["test_number"][number] - 1]:
        info = "\t-\t SUCCESS"
      elif rate <= error_from_paper[parameters["test_number"][number] - 1] * (1 + tolerance_to_paper):
        info = "\t-\t in tolerance"
      else:
        info = ""

      print("\t[%d] Test %d: %0.2f%% (paper: %0.2f%%) \t-\t Duration: %d sec %s" % (number+1,
                                                                                    parameters["test_number"][number],
                                                                                    rate,
                                                                                    error_from_paper[parameters["test_number"][number] - 1],
                                                                                    duration,
                                                                                    info))
    print("=" * 40)
    print("Total duration: %d seconds" % (time.time() - start_time_tests))

  else:
    print("\nATTENTION: The parameter dictionary is not filled correctly!")

  plt.show()
  print("\ndone")