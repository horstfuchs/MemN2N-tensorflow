import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from memN2N_data import Data

# =================================================================================================
# End-To-End Memory Network
# =================================================================================================
class MemN2N:
  def __init__(self, dictionary_size, memory_size, batch_size=32, embedding_dimensions=20,
               number_hops=3, position_encoding=False, embedding_lookup=False, linear_start=False,
               std_deviation=0.1, max_sentence_length=13, epochs_lr_annealing=25):
    """
    Initializes the End-To-End Memory Network

    :param dictionary_size: The number of different words stored in the dictionary.
    :param memory_size: The number of sentences the memory can hold.
    :param batch_size: The number of questions which are processed in one training step.
    :param embedding_dimensions:  Each word is transformed in a vector of (d x 1). This is called embedding.
    :param number_hops: Number of hops/layers of the memory network.
    :param position_encoding: If True, the network uses Position Encoding, i.e., the order of the words in the sentence
                              is taken in account.
                              NOTE: this only works in combination with embedding_lookup=True.
    :param embedding_lookup: if True, the network uses the function tf.embedding_lookup() for embedding,
                             otherwise it uses a simple matrix multiplication.
                             NOTE: make sure that you process the data according to this setting.
    :param linear_start: if True, the network uses the Linear Start method mentioned in the paper.
    :param std_deviation: standard deviation of the initialization of the embedding matrices
    :param max_sentence_length: Maximal number of words in a sentence
    :param epochs_lr_annealing: Number of epochs after which the learning rate is annealed by a factor of 0.5 (until epoch 100)
    """
    print("USING: Layer wise (RNN-like) weight tying. \n")

    self.debug_output = True
    if self.debug_output:
      self.p_0 = None
      self.p_0_array = None
      self.m_i = None
      self.p_1 = None
      self.p_1_array = None
      self.p_2 = None
      self.p_2_array = None

    # Settings of the memory network
    self.embedding_dimensions = embedding_dimensions
    self.dictionary_size = dictionary_size
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.number_hops = number_hops
    self.position_encoding = position_encoding
    self.embedding_lookup = embedding_lookup
    self.linear_start = linear_start
    self.epochs_lr_annealing = epochs_lr_annealing

    # Timestamp for entries in TensorBoard
    self.timestamp = time.strftime('%Y-%m-%d--%H-%M-%S')

    # Initialization of variables used in multiple functions
    self.summaries_directory = "summaries"
    self.sess = None
    self.predicted_answer = None
    self.eval_correct = None
    self.trainOp = None
    self.training_loss_summary = None
    self.validation_loss_summary = None
    self.clipped_grads_and_vars = None
    self.loss = None

    self.max_sentence_length = max_sentence_length

    # Placeholders
    with tf.name_scope("Input"):
      self.a = tf.placeholder(tf.int32, shape=[None], name="knownAnswer")
      if not self.embedding_lookup:
        self.q = tf.placeholder(tf.float32, shape=[None, self.dictionary_size], name="question")
        self.x = tf.placeholder(tf.float32, shape=[None, self.dictionary_size, self.memory_size], name="sentences")
      else:
        self.q = tf.placeholder(tf.int32, shape=[None, self.max_sentence_length], name="question")
        self.x = tf.placeholder(tf.int32, shape=[None, self.memory_size, self.max_sentence_length], name="sentences")

    self.learning_rate = tf.placeholder(tf.float32, shape=[], name="LearningRate")
    self.linear_training_active = tf.placeholder(tf.bool, shape=[], name="linearTrainingActive")

    d = self.embedding_dimensions
    V = self.dictionary_size
    M = self.memory_size

    # Initialization of the embedding matrices
    with tf.name_scope("Embedding_Matrices"):
      if not self.embedding_lookup:
        self.emb_matrices = tf.Variable(tf.random_normal([4, 1, d, V], stddev=std_deviation), name="ABCW")
      else:
        # the first word in the dictionary is the nil word
        nil_word_slot = tf.zeros([4, 1, 1, d])
        random_initialization = tf.random_normal([4, 1, V-1, d], stddev=std_deviation)
        emb_matrices = tf.concat(2, (nil_word_slot, random_initialization))
        self.emb_matrices = tf.Variable(emb_matrices, name="AC")
        self.position_encoding_matrix = tf.constant(self.positionEncoding(), name="PE")

    self.matrix_H = tf.Variable(tf.random_normal([embedding_dimensions, embedding_dimensions], stddev=std_deviation), name="H")

    # Initialization of the temporal matrices
    with tf.name_scope("Temporal_Matrices"):
      if not self.embedding_lookup:
        self.temporal_matrices = tf.Variable(tf.random_normal([2, 1, d, M], stddev=std_deviation), name="T_AC")
      else:
        self.temporal_matrices = tf.Variable(tf.random_normal([2, 1, M, d], stddev=std_deviation), name="T_AC")


  def positionEncoding(self):
    """
    Calculates the matrix for position encoding (using embedding_lookup)

    :return: the postions encoding matrix 'l'
    """
    matrix = np.ones((self.embedding_dimensions, self.max_sentence_length), dtype=np.float32)
    for k in range(1, self.embedding_dimensions + 1):
      for j in range(1, self.max_sentence_length + 1):
        matrix[k-1, j-1] = (1 - j / self.max_sentence_length) - (k / self.embedding_dimensions) * (1 - 2 * j / self.max_sentence_length)

    return np.transpose(matrix)


  def buildGraph(self):
    """
    Build the TensorFlow Graph for training and testing

    :return: Nothing
    """
    with tf.name_scope("Embedding"):
      A = self.emb_matrices[0]
      B = self.emb_matrices[1, 0]
      C = self.emb_matrices[2]

      if not self.embedding_lookup:
        W = tf.transpose(self.emb_matrices[3, 0], name="weights")
        u = tf.matmul(B, self.q, transpose_b=True, name="embeddedQuestion")
      else:
        W = self.emb_matrices[3, 0]
        q_emb = tf.nn.embedding_lookup(B, self.q)
        if self.position_encoding:
          position_encoding = tf.tile(tf.reshape(self.position_encoding_matrix, (1, self.max_sentence_length, self.embedding_dimensions)), (self.batch_size, 1, 1))
          q_emb = tf.mul(position_encoding, q_emb)
          # q_emb = self.position_encoding_matrix * q_emb
        u = tf.transpose(tf.reduce_sum(q_emb, 1),  name="embeddedQuestion")

    for hop in range(self.number_hops):
      # Different approach with external time matrix
      # m = tf.batch_matmul(self.emb_matrices[hop], self.x, name="embeddedSentencesForInput") + tf.mul(self.temporal_matrices[hop], time_matrix_cut)
      # c = tf.batch_matmul(self.emb_matrices[hop + 1], self.x, name="embeddedSentencesForOutput") + tf.mul(self.temporal_matrices[hop + 1], time_matrix_cut)

      if not self.embedding_lookup:
        A_batches = tf.tile(A, (self.batch_size, 1, 1), name="A_batches")
        C_batches = tf.tile(C, (self.batch_size, 1, 1), name="C_batches")

        m = tf.batch_matmul(A_batches, self.x, name="embeddedSentencesForInput") + self.temporal_matrices[0]
        c = tf.batch_matmul(C_batches, self.x, name="embeddedSentencesForOutput") + self.temporal_matrices[1]

      else:
        a_emb = tf.nn.embedding_lookup(A[0], self.x, name="embeddingLookup_A")
        if self.position_encoding:
          position_encoding = tf.tile(tf.reshape(self.position_encoding_matrix, (1, 1, self.max_sentence_length, self.embedding_dimensions)), (self.batch_size, self.memory_size, 1, 1))
          a_emb = tf.mul(position_encoding, a_emb, name="positionEncoding_A")
          # a_emb = self.position_encoding_matrix * a_emb
        m = tf.reduce_sum(a_emb, 2) + self.temporal_matrices[0, 0]
        m = tf.transpose(m, (0, 2, 1), name="memory")

        c_emb = tf.nn.embedding_lookup(C[0], self.x, name="embeddingLookup_C")
        if self.position_encoding:
          position_encoding = tf.tile(tf.reshape(self.position_encoding_matrix, (1, 1, self.max_sentence_length, self.embedding_dimensions)), (self.batch_size, self.memory_size, 1, 1))
          c_emb = tf.mul(position_encoding, c_emb, name="positionEncoding_C")
          # c_emb = self.position_encoding_matrix * c_emb
        c = tf.reduce_sum(c_emb, 2) + self.temporal_matrices[1, 0]
        c = tf.transpose(c, (0, 2, 1), name="output")

      with tf.name_scope("InputMemoryRepresentation"):
        u_resh = tf.reshape(tf.transpose(u), (self.batch_size, 1, -1), name="reshapedEmbeddedQuestion")
        p_lin = tf.batch_matmul(u_resh, m, name="linearProbabilities")

        # Functions for the Linear Start method
        def linearTraining(): return p_lin
        def nonlinearTraining(): return tf.nn.softmax(p_lin, name="SoftmaxInput")
        p = tf.cond(self.linear_training_active, linearTraining, nonlinearTraining, name="linearOrNonlinearTraining")

      with tf.name_scope("OutputMemoryRepresentation"):
        o = tf.batch_matmul(p, c, adj_y=True, name="WeightedSumOutput")

      with tf.name_scope("SummationPrediction"):
        o_2d = tf.reshape(o, (self.batch_size, self.embedding_dimensions), name="reshape_o")
        u = tf.add(tf.transpose(o_2d), tf.matmul(self.matrix_H, u), name="sum_u")

      if self.debug_output:
        if hop == 0:
          self.p_0 = tf.nn.top_k(p, 1)
          self.p_0_array = p
          self.m_i = m
        elif hop == 1:
          self.p_1 = tf.nn.top_k(p, 1)
          self.p_1_array = p
        else:
          self.p_2 = tf.nn.top_k(p, 1)
          self.p_2_array = p

    with tf.name_scope("FinalPrediction"):
      # a_hat = tf.nn.softmax(tf.transpose(tf.matmul(W, (o + u))), name="predictedAnswer") # reason: sparse_softmax_cross... performs internally a softmax
      a_hat = tf.transpose(tf.matmul(W, u), name="predictedAnswer")

      predicted_probabilities = tf.nn.softmax(a_hat) #log_softmax

      self.predicted_answer = tf.argmax(predicted_probabilities, dimension=1)
      self.eval_correct = tf.nn.in_top_k(predicted_probabilities, self.a, 1, name="evalCorrect")


    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(a_hat, self.a, name='xEntropy')
    self.loss = tf.reduce_sum(cross_entropy, name='xEntropy_mean') # using sum not mean (like in the paper suggested)

    # Summaries
    self.training_loss_summary = tf.scalar_summary("training_crossEntropy_loss", self.loss)
    self.validation_loss_summary = tf.scalar_summary("validation_crossEntropy_loss", self.loss)

    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    with tf.name_scope("Minimize"):
      clipping = True
      if not clipping:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.trainOp = optimizer.minimize(self.loss, global_step=global_step, name="minimize")
      else:
        max_norm = 40

        # Compute the gradients for a list of variables.
        parameters = [self.emb_matrices, self.temporal_matrices]
        grads_and_vars = optimizer.compute_gradients(self.loss, parameters)

        if not self.embedding_lookup:
          self.clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], max_norm), gv[1]) for gv in grads_and_vars]
          self.trainOp = optimizer.apply_gradients(self.clipped_grads_and_vars, name="applyGradients")
        else:
          # Make sure that the nil word is not learned
          nil_zeros = tf.zeros((self.number_hops+1, 1, 1, self.embedding_dimensions))
          fixed_grads_and_vars = []
          is_emb_matrix = True # is first row
          for g, v in grads_and_vars:
            if is_emb_matrix:
              g = tf.concat(2, (nil_zeros, g[:, :, 1:]))
              is_emb_matrix = False

            fixed_grads_and_vars.append((g, v))

          self.clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], max_norm, name="clipGradients"), gv[1]) for gv in fixed_grads_and_vars]
          self.trainOp = optimizer.apply_gradients(self.clipped_grads_and_vars, name="applyGradients")


  def train(self, sentences, questions, answers, initial_learning_rate=0.01, number_epochs=60, data=None):
    """
    Training of the End-To-End Memory Network.

    :param sentences: matrix containing all sentences for training
    :param questions: matrix containing all questions for training
    :param answers: matrix containing all answers for training
    :param initial_learning_rate: the initial learning rate (learning rate anneals every 15 epochs by 1/2.
    :param number_epochs: number of training epochs
    :param data: object containing the Data class used for processing the data (for inverse lookup)
    :return: nothing
    """
    # Session for training, validation and testing (is closed after testing)
    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

    # Create training and validation set
    size_validation_set = 0.1 # 10% validation set

    learning_rate = initial_learning_rate
    number_batches = answers.shape[0]
    number_training_batches = int((1 - size_validation_set) * number_batches)

    training_questions = questions[0:number_training_batches]
    training_sentences = sentences[0:number_training_batches]
    training_answers   = answers[0:number_training_batches]

    validation_questions = questions[number_training_batches:]
    validation_sentences = sentences[number_training_batches:]
    validation_answers   = answers[number_training_batches:]

    # Change the encoding to all ones (BoW) if not using embedding_lookup()
    training_questions_encoded = self.matrixEncoding(training_questions, self.embedding_lookup, self.position_encoding)
    training_sentences_encoded = self.matrixEncoding(training_sentences, self.embedding_lookup, self.position_encoding)

    validation_questions_encoded = self.matrixEncoding(validation_questions, self.embedding_lookup, self.position_encoding)
    validation_sentences_encoded = self.matrixEncoding(validation_sentences, self.embedding_lookup, self.position_encoding)

    # Train writer for TensorBoard
    train_writer = tf.train.SummaryWriter(self.summaries_directory + '/{}'.format(self.timestamp) + '/train', self.sess.graph)

    training_error_list = []
    validation_error_list = []
    previous_validation_loss_value = np.inf

    for epoch in range(number_epochs):
      # Annealing of the learning rate
      if epoch > 0 and epoch % self.epochs_lr_annealing == 0 and epoch <= 100:
        learning_rate *= 0.5

      # TRAINING
      training_batch_print = [training_sentences, training_questions]
      training_error_rate, training_loss_value = self.trainOrValidate(training_sentences_encoded,
                                                                      training_questions_encoded,
                                                                      training_answers,
                                                                      is_training=True,
                                                                      epoch=epoch,
                                                                      batch_for_print=training_batch_print,
                                                                      train_writer=train_writer,
                                                                      learning_rate=learning_rate,
                                                                      data=data,
                                                                      is_linear_model=self.linear_start)
      training_error_list.append(training_error_rate)

      # VALIDATION
      validation_batch_print = [validation_sentences, validation_questions]
      if self.linear_start:
        validation_error_rate, validation_loss_value = self.trainOrValidate(validation_sentences_encoded,
                                                                            validation_questions_encoded,
                                                                            validation_answers,
                                                                            is_training=False,
                                                                            epoch=epoch,
                                                                            batch_for_print=validation_batch_print,
                                                                            data=data,
                                                                            train_writer=train_writer)

        if epoch > 0 and validation_loss_value > previous_validation_loss_value:
          if self.linear_start:
            print("=======> Switch to NONLINEAR training!")
          self.linear_start = False

        previous_validation_loss_value = validation_loss_value
        print("Epoch %d: training loss = %.6f, validation loss = %.6f" % (epoch, training_loss_value, validation_loss_value))

      else:
        validation_error_rate, _ = self.trainOrValidate(validation_sentences_encoded,
                                                        validation_questions_encoded,
                                                        validation_answers,
                                                        is_training=False,
                                                        epoch=epoch,
                                                        batch_for_print=validation_batch_print,
                                                        data=data)
        print("Epoch %d: loss = %.6f" % (epoch, training_loss_value))

      validation_error_list.append(validation_error_rate)

      # Print error for each episode
      print("\ttraining error = %0.3f%%, validation error = %0.3f%%" % (training_error_list[-1], validation_error_list[-1]))

    train_writer.close()

    # PLOT figure of training and validation error
    self.plotErrors(training_error_list, validation_error_list)


  def test(self, sentences, questions, answers, data):
    """
    Testing function of the End-To-End Memory Network

    :param sentences: contains all sentences
    :param questions: contains all questions
    :param answers: contains all correct answers
    :param data: the Data object which was used to process the training data (used for inverse lookup of words)
    :return: error rate
    """
    questions_encoded = self.matrixEncoding(questions, self.embedding_lookup, self.position_encoding)
    sentences_encoded = self.matrixEncoding(sentences, self.embedding_lookup, self.position_encoding)

    batch_to_print = [sentences, questions]
    error_rate, _ = self.trainOrValidate(sentences_encoded,
                                         questions_encoded,
                                         answers,
                                         is_training=False,
                                         batch_for_print=batch_to_print,
                                         data=data)
    print('=> TEST error rate: %0.03f%%' % error_rate)

    return error_rate


  def trainOrValidate(self, sentences_encoded, questions_encoded, answers, is_training=False, epoch=0, batch_for_print=None, train_writer=None, learning_rate=0, data=None, is_linear_model=False):
    """
    Provides the functionality to train and validate the network. It can also be used for testing (is_training=False).

    It also stores training information in TensorBoard as well as it prints debug output.

    :param sentences_encoded: contains all sentences
    :param questions_encoded: contains all questions
    :param answers: contains all answers
    :param is_training: if True, training is performed on this dataset,
                        otherwise validation/testing
    :param epoch: number of the current training epoch (for output)
    :param batch_for_print: If specified, this batch is used for printing the probabilities that a sentence
                            contains significant information
    :param train_writer: the TensorFlow train writer to store training data
    :param learning_rate: the current learning rate used for this training step
    :param data: the Data object which was used to process the training data (used for inverse lookup of words)
    :param is_linear_model: if True, the training uses Linear Start
    :return: the error rate and the loss
    """
    number_batches = answers.shape[0]
    number_correct_answers = 0
    loss_value = np.NaN

    mean_validation_loss_list = np.inf * np.ones(number_batches)
    for step in range(number_batches):
      # time_matrices = self.createTimeMatrix(sentences[step])

      feed_dictionary = {
          self.a: answers[step],
          self.q: questions_encoded[step],
          self.x: sentences_encoded[step],
          self.learning_rate: learning_rate,
          self.linear_training_active: is_linear_model#,
          # self.time_matrix_cut: time_matrices
      }

      if step < 1 and epoch in [0, 29, 59]:
        if is_training:
          self.saveTensorBoardImages(epoch, train_writer, feed_dictionary)

          # plot gradient values into histograms
          gradient_summary = tf.histogram_summary("grad_hist_{:03d}".format(epoch), self.clipped_grads_and_vars[0][0])
          gradient_summary_str = self.sess.run(gradient_summary, feed_dict=feed_dictionary)
          train_writer.add_summary(gradient_summary_str)

        if batch_for_print is not None and self.debug_output:
          if is_training:
            print("=" * 25 + "\nTraining")
          else:
            print("=" * 25 + "\nValidation/Testing")
          self.printProbsWithSentences(batch_for_print, answers, data, step, feed_dictionary)

      if is_training:
        output_eval, _, loss_value, loss_summary_str = self.sess.run([self.eval_correct, self.trainOp, self.loss, self.training_loss_summary], feed_dict=feed_dictionary)
        train_writer.add_summary(loss_summary_str, epoch * number_batches + step)
      elif self.linear_start and train_writer is not None:
        output_eval, val_loss_value, loss_summary_str = self.sess.run([self.eval_correct, self.loss, self.validation_loss_summary], feed_dict=feed_dictionary)
        train_writer.add_summary(loss_summary_str, epoch * number_batches + step)
        mean_validation_loss_list[step] = val_loss_value
      else:
        output_eval, loss_value = self.sess.run([self.eval_correct, self.loss], feed_dict=feed_dictionary)

      number_correct_answers += np.sum(output_eval)

    if self.linear_start and not is_training:
      loss_value = np.mean(mean_validation_loss_list)

    error_rate = self.calculateError(number_correct_answers, number_batches)

    return error_rate, loss_value


  def calculateError(self, number_correct_answers, number_batches):
    """
    Calculates the error rate over the total number of answers

    :param number_correct_answers: number of correct answers
    :param number_batches: number of batches used
    :return: the error rate
    """
    return (1 - number_correct_answers / (self.batch_size * number_batches)) * 100


  def plotErrors(self, training_error_list, validation_error_list):
    """
    Plots the training and validation errors over the epochs in a figure

    :param training_error_list: list containing all training error rates
    :param validation_error_list: list containing all validation error rates
    :return: nothing
    """
    plt.figure()
    plt.plot(training_error_list)
    plt.plot(validation_error_list)
    plt.xlabel("epochs")
    plt.ylabel("error in %")
    plt.legend(["training", "validation"])
    plt.draw()


  def processTensorBoardColourImage(self, matrix):
    """
    Creates a color image for TensorBoard,
    where
      - Red indicates negative values and
      - Green indicates positive values.

    :param matrix: matrix that should be processed
    :return: a matrix with the three color channels
    """
    matrix_pos = matrix
    matrix_neg = matrix
    matrix_zeros = tf.zeros_like(matrix)

    matrix_neg = tf.select(matrix_neg > 0, matrix_zeros, matrix_neg)
    matrix_neg = tf.abs(matrix_neg)

    matrix_pos = tf.select(matrix_pos < 0, matrix_zeros, matrix_pos)
    matrix_pos = tf.abs(matrix_pos)

    return tf.concat(3, (matrix_neg, matrix_pos, matrix_zeros))


  def saveTensorBoardImages(self, epoch, train_writer, feed_dictionary):
    """
    Store embedding matrices, temporal matrices, and gradients as images in TensorBoard.

    :param epoch: current epoch (used to name the images)
    :param train_writer: train writer of TensorFlow
    :param feed_dictionary: current feed_dictionary
    :return: nothing
    """
    if not self.embedding_lookup:
      emb_matrices_img = self.processTensorBoardColourImage(tf.reshape(self.emb_matrices, (4, self.embedding_dimensions, self.dictionary_size, 1)))
    else:
      emb_matrices_img = self.processTensorBoardColourImage(tf.reshape(self.emb_matrices, (4, self.dictionary_size, self.embedding_dimensions, 1)))
    emb_matrix_summary = tf.image_summary("emb_{:03d}".format(epoch), emb_matrices_img, max_images=self.number_hops + 1)
    emb_matrix_summary_str = self.sess.run(emb_matrix_summary, feed_dict=feed_dictionary)
    train_writer.add_summary(emb_matrix_summary_str, epoch)

    memories_to_print = self.memory_size
    if not self.embedding_lookup:
      temp_matrices_img = self.processTensorBoardColourImage(tf.reshape(self.temporal_matrices[:, :, :, 0:memories_to_print], (2, self.embedding_dimensions, memories_to_print, 1)))
    else:
      temp_matrices_img = self.processTensorBoardColourImage(tf.reshape(self.temporal_matrices[:, :, 0:memories_to_print], (2, memories_to_print, self.embedding_dimensions, 1)))
    temp_matrix_summary = tf.image_summary("temp_{:03d}".format(epoch), temp_matrices_img, max_images=self.number_hops + 1)
    temporal_matrix_summary_str = self.sess.run(temp_matrix_summary, feed_dict=feed_dictionary)
    train_writer.add_summary(temporal_matrix_summary_str, epoch)

    gradient_img = self.processTensorBoardColourImage(tf.reshape(self.clipped_grads_and_vars[0][0], (4, self.dictionary_size, self.embedding_dimensions, 1)))
    gradients_summary = tf.image_summary("grad_{:03d}".format(epoch), gradient_img, max_images=4)
    gradients_summary_str = self.sess.run(gradients_summary, feed_dict=feed_dictionary)
    train_writer.add_summary(gradients_summary_str, epoch)


  def printProbsWithSentences(self, batch_for_print, answers, data, step, feed_dictionary):
    """
    Prints the most probable sentences for each hop

    :param batch_for_print: batch of sentences and questions for printing
    :param answers: correct answers
    :param data: Data object that was used to process the data (for inverse lookup of words)
    :param step: current training step
    :param feed_dictionary: current feed_dictionary
    :return: nothing
    """
    sentences = batch_for_print[0]
    questions = batch_for_print[1]

    output_predicted_answers = None
    p0_val = None
    p1_val = None
    p2_val = None

    if self.number_hops == 3:
      output_predicted_answers, p0_val, p1_val, p2_val = self.sess.run(
              [self.predicted_answer, self.p_0, self.p_1, self.p_2], feed_dict=feed_dictionary)
    if self.number_hops == 2:
      output_predicted_answers, p0_val, p1_val = self.sess.run(
              [self.predicted_answer, self.p_0, self.p_1], feed_dict=feed_dictionary)
    if self.number_hops == 1:
      output_predicted_answers, p0_val = self.sess.run(
              [self.predicted_answer, self.p_0], feed_dict=feed_dictionary)

    print("-----")
    print("Batch Nr. %d" % step)
    number_printed_stories = 5 if (self.batch_size >= 5) else self.batch_size
    for i in range(number_printed_stories):
      print("-----")
      if not self.embedding_lookup:
        print("%s?" % data.lookupBagOfWord(questions[step, i]))
      else:
        print("%s?" % data.lookupBagOfWord_embeddingLookup(questions[step, i]))

      p0_val = np.reshape(np.array(p0_val), [2, -1])
      if not self.embedding_lookup:
        print("Hop 1: p=%0.3f: [%d] %s." % (p0_val[0, i], p0_val[1, i], data.lookupBagOfWord(sentences[step, i, :, int(p0_val[1, i])])))
      else:
        print("Hop 1: p=%0.3f: [%d] %s." % (p0_val[0, i], p0_val[1, i], data.lookupBagOfWord_embeddingLookup(sentences[step, i, int(p0_val[1, i])])))

      if self.number_hops >= 2:
        p1_val = np.reshape(np.array(p1_val), [2, -1])
        if not self.embedding_lookup:
          print("Hop 2: p=%0.3f: [%d] %s." % (p1_val[0, i], p1_val[1, i], data.lookupBagOfWord(sentences[step, i, :, int(p1_val[1, i])])))
        else:
          print("Hop 2: p=%0.3f: [%d] %s." % (p1_val[0, i], p1_val[1, i], data.lookupBagOfWord_embeddingLookup(sentences[step, i, int(p1_val[1, i])])))

      if self.number_hops == 3:
        p2_val = np.reshape(np.array(p2_val), [2, -1])
        if not self.embedding_lookup:
          print("Hop 3: p=%0.3f: [%d] %s." % (p2_val[0, i], p2_val[1, i], data.lookupBagOfWord(sentences[step, i, :, int(p2_val[1, i])])))
        else:
          print("Hop 3: p=%0.3f: [%d] %s." % (p2_val[0, i], p2_val[1, i], data.lookupBagOfWord_embeddingLookup(sentences[step, i, int(p2_val[1, i])])))

      print("Correct: %s, \t Predicted: %s" % (data.inverseLookup(answers[step][i]), data.inverseLookup(output_predicted_answers[i])))


  def matrixEncoding(self, matrix, embedding_lookup, position_encoding=False):
    """
    Encodes the matrix for the version which does not use embedding_lookup.

    FIXME: add Position Encoding (PE)

    :param matrix: matrix to encode
    :param position_encoding: if True Position Encoding for the simple matrix multiplication embedding is used.
                              TODO not implemented
    :return: encoded matrix
    """
    if not embedding_lookup:
      encoded_matrix = matrix.copy()
      encoded_matrix[encoded_matrix > 0] = 1

      if position_encoding:
        print("ATTENTION: Position encoding with simple matrix multiplication embedding NOT implemented!")

    else:
      encoded_matrix = matrix

    return encoded_matrix


  def deleteSummaryFiles(self):
    """
    Deletes the summary files that are used by TensorBoard

    :return:
    """
    if tf.gfile.Exists(self.summaries_directory):
      tf.gfile.DeleteRecursively(self.summaries_directory)

    tf.gfile.MakeDirs(self.summaries_directory)


  def closeSession(self):
    """
    Close the TensorFlow session

    :return: nothing
    """
    self.sess.close()


  # def createTimeMatrix(self, sentences_batch):
  #   """
  #   Create a time matrix as a mask for the temporal matrix T_A and T_C
  #
  #   :param sentences_batch:
  #   :return:
  #   """
  #   number_sentences = np.sum(sentences_batch, axis=1) > 0
  #
  #   time_matrix = np.arange(1, self.memory_size + 1)
  #   time_matrices = np.copy(time_matrix)
  #   time_matrices[np.invert(number_sentences[0])] = 0
  #
  #   for i in range(1, self.batch_size):
  #     cut_time_matrix = np.copy(time_matrix)
  #     cut_time_matrix[np.invert(number_sentences[i])] = 0
  #     time_matrices = np.vstack([time_matrices, cut_time_matrix])
  #
  #   return np.reshape(time_matrices, (self.batch_size, 1, self.memory_size))