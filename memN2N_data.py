import numpy as np

def splitNumberAndText(line):
  """
  Split one text line

  :param line: text line of the input file
  :return: number of the sentence and the sentence string
  """
  numberAndText = line.split(' ', maxsplit=1)
  sentenceNumber = numberAndText[0]
  text = numberAndText[1]

  return sentenceNumber, text[:-1]


def extractText(text):
  """
  Extract the text of the text line.

  Structure:
  if sentence: text line only contains the sentence
  if question: text = "<question>\t<answer>\t<supporting sentences>"

  :param text: text line of input
  :return: - the sentence or a list with question/answer/supporting sentences
           - isQuestion: True if 'text' contains a question, otherwise False
  """
  isQuestion = False
  splitText = text.split('\t')

  if len(splitText) > 1:
    isQuestion = True

  return splitText, isQuestion


# ========================================================================
# Data class for preprocessing of the training/testing files
# ========================================================================
class Data:
  def __init__(self, embedding_lookup):
    """
    Init function of the Data class

    :param embedding_lookup: if True the class generates the data for being used with
                             the TensorFlow method tf.embedding_lookup(),
                             otherwise it generates data which can be embedded with 7
                             simple matrix multiplications
    """
    self.embedding_lookup = embedding_lookup
    self.dictionary = dict()
    self.dictionary["null-symbol"] = 0
    self.dictionary_size = 1 # 0 is no word
    self.inverse_dictionary = dict()


  def loadData(self, filename):
    """
    Load the text from a file

    :param filename: name of the file (must contain also the file path if the file is in a different folder)
    :return: a list with stories
    """
    file = open(filename, 'r')

    stories = []
    story = []
    prev_number = 0

    for line in file:
      # Extract line number of sentence/question
      number, text = splitNumberAndText(line)
      number = int(number)

      # Store previous story
      if prev_number > number:
        stories.append(story)
        story = []

      text, is_question = extractText(text)

      story.append([number, is_question, text])
      prev_number = number

    file.close()

    return stories


  def translate(self, words):
    """
    Translates the word into an integer using a word dictionary.
    This dictionary is created on the fly while checking if words are already in the dictionary.

    :param words: list of words to translate
    :return: a list with the translated words into integers
    """
    translated_words = []
    for word in words:
      word = word.lower()
      if word in self.dictionary:
        translated_words.append(self.dictionary[word])
      else:
        # print("add " + word)
        self.dictionary[word] = self.dictionary_size
        translated_words.append(self.dictionary_size)
        self.dictionary_size += 1

    return translated_words


  def translateWord(self, word=""):
    return self.translate([word])


  def translateSentence(self, sentence=""):
    sentence = sentence.strip()
    sentence = sentence[:-1] # remove the punctuation mark
    words = sentence.split(" ")

    return self.translate(words)


  def createStoryMatrix(self, stories, memory_size):
    """
    Create a matrix holding all stories/sentences.
    The translation of the word, i.e., the integer, is used to determine the position in the matrix.
    Additionally, the position of the word in the sentence is stored in this matrix (starting to count with 1).

    :param stories: All the stories
    :param memory_size: The number of sentences the memory can hold.
    :return: matrix of size (number stories x dictionary size x memory size)
    """
    matrix = np.zeros([len(stories), self.dictionary_size, memory_size])

    story_idx = 0
    for story in stories:

      number_sentences = len(story)
      if number_sentences > memory_size:
        first_sentence = number_sentences - memory_size
        memory_idx = memory_size - 1
      else:
        first_sentence = 0
        memory_idx = number_sentences - 1

      sentence_idx = 0
      for sentence in story:
        if sentence_idx >= first_sentence:
          for word_idx, word_pos in zip(sentence, range(1, len(sentence) + 1)):
            matrix[story_idx, word_idx, memory_idx] = word_pos
          memory_idx -= 1
        sentence_idx += 1
      story_idx += 1

    return matrix


  def createQuestionMatrix(self, questions):
    """
    Create a matrix holding all questions.
    The translation of the word, i.e., the integer, is used to determine the position in the matrix.
    Additionally, the position of the word in the question is stored in this matrix (starting to count with 1).

    :param questions: All the questions
    :return: matrix of size (number questions x dictionary size)
    """
    matrix = np.zeros([len(questions), self.dictionary_size])

    question_idx = 0
    for question in questions:
      for word_idx, word_pos in zip(question, range(1, len(question) + 1)):
        matrix[question_idx, word_idx] = word_pos
      question_idx += 1

    return matrix


  def createAnswerMatrix(self, answers):
    """
    Creat a matrix holding all answers.
    The translation of the word, i.e., the integer, is used to determine the position in the matrix.

    :param answers: All the answers
    :return: matrix of size (number answers x dictionary size)
    """
    matrix = np.zeros([len(answers), self.dictionary_size])

    answer_idx = 0
    for word_idx in answers:
      matrix[answer_idx, word_idx] = 1
      answer_idx += 1

    return matrix


  def processStories(self, stories):
    """
    Translate all sentences/questions/answers to integers.

    :param stories: Stories in textual form
    :return: translated stories, sentences and answers and in addition, the length of the longest story (i.e., most sentences)
    """
    longest_story = 0
    answers = []
    translated_questions = []
    translated_sentences = []

    for items in stories:
      sentences = []
      for item in items:
        if item[1] == False:
          sentences.append(self.translateSentence(item[2][0]))
        else:
          question = item[2][0]
          translated_questions.append(self.translateSentence(question))

          answer = item[2][1]
          answers.append(self.translateWord(answer))
          translated_sentences.append(sentences.copy())

          if len(sentences) > longest_story:
            longest_story = len(sentences)

    print("Longest story has %d sentences." % longest_story)

    return [translated_questions, translated_sentences, answers], longest_story


  def createMatrices(self, translated_stories, memory_size, batch_size, shuffle, max_sentence_length=13):
    """
    Create matrices out of the translated stories, so the neural network can process them.
    Note: In this method there are two version implemented. For using tf.embedding_lookup() we need to process
          the data different than when we use simple matrix multiplication for the embedding.

    :param translated_stories: stories, where words are represented as integers
    :param memory_size: The number of sentences the network can remember
    :param batch_size: The size of a batch that is used for training/testing
    :param shuffle: if True the stories are shuffled, otherwise they appear in the batches in the order of the input file.
    :param max_sentence_length: maximum number of words in a sentence.
    :return: the stories in batches, ready do use for the NN
    """
    embedded_questions = translated_stories[0]
    embedded_sentences = translated_stories[1]
    answers = translated_stories[2]

    # Create the question matrix
    if not self.embedding_lookup:
      question_matrix = self.createQuestionMatrix(embedded_questions)
    else:
      question_matrix = np.zeros((len(embedded_questions), max_sentence_length))
      for i in range(len(embedded_questions)):
        for k in range(len(embedded_questions[i])):
          question_matrix[i,k] = embedded_questions[i][k]

    # Create the matrix containing all the sentences
    if not self.embedding_lookup:
      sentences_matrix = self.createStoryMatrix(embedded_sentences, memory_size=memory_size)
    else:
      sentences_matrix = np.zeros((len(embedded_questions), memory_size, max_sentence_length))
      for i in range(len(embedded_sentences)):
        number_of_sentences = len(embedded_sentences[i])
        if number_of_sentences > memory_size:
          start = number_of_sentences - memory_size
          mem_idx = memory_size - 1
        else:
          start = 0
          mem_idx = number_of_sentences - 1
        for j in range(start, number_of_sentences):
          for k in range(len(embedded_sentences[i][j])):
            sentences_matrix[i,mem_idx,k] = embedded_sentences[i][j][k]
          mem_idx -= 1

    # Create the vector containing the answers
    answer_matrix = np.array(answers)
    # answer_matrix = self.createAnswerMatrix(answers)

    if shuffle:
      sentences_matrix, question_matrix, answer_matrix = self.shuffleTrainingData(sentences_matrix, question_matrix, answer_matrix)

    batches_sentences, batches_questions, batches_answers = self.createBatches(sentences_matrix, question_matrix, answer_matrix, batch_size, memory_size)

    return batches_sentences, batches_questions, batches_answers


  def createBatches(self, sentences_matrix, question_matrix, answer_matrix, batch_size, memory_size):
    """
    Creates batches for batch learning.

    :param sentences_matrix: containing a matrix with all sentences
    :param question_matrix: containing a matrix with all questions
    :param answer_matrix: containing a matrix with all answers
    :param batch_size: number of stories (i.e. sentences/questions/answers) in one batch
    :param memory_size: maximal number of sentences the memory network can remember
    :return: matrices with the batches containing the sentences/questions/answers
    """
    number_questions = question_matrix.shape[0]
    number_batches = int(number_questions / batch_size)

    batches_questions = np.reshape(question_matrix[0:number_batches*batch_size,:], (number_batches, batch_size, -1))
    batches_answers = np.reshape(answer_matrix[0:number_batches*batch_size], (number_batches, batch_size))

    if not self.embedding_lookup:
      batches_sentences = np.reshape(sentences_matrix[0:number_batches*batch_size], (number_batches, batch_size, self.dictionary_size, -1))
    else:
      batches_sentences = np.reshape(sentences_matrix[0:number_batches*batch_size], (number_batches, batch_size, memory_size, -1))

    return batches_sentences, batches_questions, batches_answers


  def createInverseDictionary(self):
    """
    create an second dictionary for inverse dictionary lookups, i.e., "integer to word" translations

    :return: Nothing
    """
    self.inverse_dictionary = {}
    for k in self.dictionary:
      self.inverse_dictionary[self.dictionary[k]] = k


  def inverseLookup(self, idx):
    """
    Inverse lookup in the inverse dictionary

    :param idx: integer that represents the word
    :return: a string containing the word
    """
    return self.inverse_dictionary[idx]


  def shuffleTrainingData(self, sentences_matrix, question_matrix, answer_matrix):
    """
    Shuffle the order of the sentences/questions/answers in the matrices.

    :param sentences_matrix: matrix containing all sentences
    :param question_matrix: matrix containing all questions
    :param answer_matrix: matrix containing all questions
    :return: the same matrices, but with shuffled data
    """
    number_questions = question_matrix.shape[0]
    perm = np.random.permutation(number_questions)

    sentences_matrix = sentences_matrix[perm]
    question_matrix = question_matrix[perm]
    answer_matrix = answer_matrix[perm]

    return sentences_matrix, question_matrix, answer_matrix


  def lookupBagOfWord(self, sentence_vector):
    """
    Translate a bag of words vector to a string with the words. It returns the words
    in the order they appeared in the sentence.
    This method is used for the simple matrix multiplication method of embedding.

    :param sentence_vector: a vector (dictionary size x 1) containing the information of the words.
    :return: the string with the words/sentence.
    """
    bag_of_words = ""

    len_of_sentence = int(max(sentence_vector))
    for word_number in range(1, len_of_sentence+1):
      for j in range(sentence_vector.shape[0]):
        if sentence_vector[j] == word_number:
          bag_of_words += self.inverse_dictionary[j] + " "

    return bag_of_words


  def lookupBagOfWord_embeddingLookup(self, sentence):
    """
    Translates a bag of words (list with integers) to a string containing the words/sentence.
    This method is used for the embedding method which uses tf.embedding_lookup().

    :param sentence: a list containing the integer representation of the words.
    :return: the string with the words/sentence.
    """
    sentence_string = ""

    for word_id in sentence:
      if sentence_string == "" or word_id != 0:
        sentence_string += self.inverse_dictionary[word_id] + " "

    return sentence_string