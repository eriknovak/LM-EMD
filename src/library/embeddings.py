import os
import ot
import operator
import numpy as np
from gensim.models import KeyedVectors, FastText
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation
from nltk.corpus import stopwords as nltk_stopwords

from .vocabulary import Vocabulary

# used for documentation
from typing import List


class Embeddings:
    def __init__(self, language, model_path, model_format="word2vec"):
        """Initializes the text embedding module
        Args:
            language (string): The language of the embedding model.
                It must be in the ISO 693-1 code format.
            model_path (string): The path to the embedding model file.
            model_format (string): The format in which the model file is stored.
                Possible options are 'word2vec' and 'fasttext'. (Default: 'word2vec')
        """
        self.__language = language

        try:
            self.__stopwords = nltk_stopwords.words(self.__language)
        except:
            self.__stopwords = []

        self.embedding = None
        self.__model = None
        # prepare the vocabulary
        self.vocabulary = Vocabulary()

        if not model_path == None:
            if os.path.isfile(model_path):
                self.__load_model(model_path, model_format)
            else:
                raise Exception(
                    "TextEmbedding.__init__: model_path does not exist {}".format(
                        model_path
                    )
                )
        else:
            raise Exception(
                "TextEmbedding.__init__: model_path does not exist {}".format(
                    model_path
                )
            )

    def __load_model(self, model_path, model_format):
        """Loads the word embedding model
        Args:
            model_path (str): The word embedding model path.
            model_format (str): The word embedding model format.
        """

        if model_format == "word2vec":
            # load the model with the word2vec format
            self.__model = KeyedVectors
            self.embedding = self.__model.load_word2vec_format(model_path)
        elif model_format == "fasttext":
            # load the model with the fasttext format
            self.__model = FastText
            self.embedding = self.__model.load_fasttext_format(model_path)
        else:
            raise Exception(
                "TextEmbedding.__load_model: Model '{}' not supported (must be 'word2vec' or 'fasttext').".format(
                    model_format
                )
                + " Cannot load word embedding model."
            )

    def get_language(self):
        """Returns the language of the text embedding model

        Returns:
            string: The ISO 693-1 code of the language.

        """

        return self.__language

    def get_stopwords(self):
        """Returns the stopwords

        Returns:
            list(string): The list of stopwords.

        """

        return self.__stopwords

    def tokenize(self, text):
        """Tokenizes the provided text

        Args:
            text (string): The text to be tokenized.

        Returns:
            list(tuple(string, number)): A list of (token, count) pairs from the text without the stopwords.

        """

        # make everything lowercase and strip punctuation
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation]
        tokens = preprocess_string(text, CUSTOM_FILTERS)

        # filter out all stopwords
        filtered_tokens = [w for w in tokens if not w in self.__stopwords]

        # count the term frequency in the text
        count = {}
        for word in filtered_tokens:
            if word not in count:
                count[word] = 0
            count[word] += 1

        # sort the terms in descending order
        terms_sorted = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
        return terms_sorted

    def add_vocabulary_corpus(self, dataset: List[str]) -> None:
        """Add a corpus to the vocabulary

        Args:
            dataset (List[str]): A list of documents/strings to be added to the vocabulary.

        """

        self.vocabulary.add_corpus(dataset)

    def add_vocabulary_document(self, document: str) -> None:
        """Add a document to the vocabulary

        Args:
            document (str): A document to be added to the vocabulary.

        """
        self.vocabulary.add_document(document)

    def word_embeddings(self, tokens: List[str]):
        """Get the word embeddings of the sentence
        """
        embeds = []
        for token in tokens:
            if token in self.embedding:
                vector = self.embedding[token]
                vector = vector / np.linalg.norm(vector)
                embeds.append(vector)
        return np.array(embeds)

    def text_embedding(
        self, text: str, weight: str = "tf_idf", normalize: bool = True
    ) -> List[float]:
        """Create the text embedding

        Args:
            text (string): The text to be embedded.

        Returns:
            list(number): The array of values representing the text embedding.

        """

        # prepare the embedding placeholder
        embeds = np.zeros(self.embedding.vector_size, dtype=np.float32)

        if text is None:
            # return the default embedding in a vanilla python object
            return embeds.tolist()

        # get the text terms with frequencies
        term_sorted = self.tokenize(text)
        # iterate through the terms and count the number of terms
        count = 0
        for token, n_appearances in term_sorted:
            # sum all token embeddings of the vector
            if token in self.embedding.vocab.keys():

                token_embedding = self.embedding[token]
                if weight == "tfidf":
                    # the token embedding is weighted with the TF-IDF value of the token in the document
                    token_embedding = (
                        token_embedding * n_appearances * self.vocabulary.get_idf(token)
                    )
                elif weight == "idf":
                    # the token embedding is weighted with the IDF value of the token in the document
                    token_embedding = token_embedding * self.vocabulary.get_idf(token)
                elif weight == "tf":
                    # the token embedding is weighted with the TF value of the token in the document
                    token_embedding = token_embedding * n_appearances

                # sum up the token embedding and increment couont
                embeds += token_embedding
                count += n_appearances

        if count == 0:
            # return the empty embedding list
            return embeds.tolist()

        if normalize:
            # normalize the embedding
            embeds /= np.linalg.norm(embeds)

        # return the embedding in vanilla python object
        return embeds.tolist()

    def wmdistance(self, text1: str, text2: str):
        # get the tokens and their appearances
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)
        # get the token distributions
        values1 = [value for key, value in tokens1]
        values2 = [value for key, value in tokens2]
        dist_text1 = np.array(values1) / sum(values1)
        dist_text2 = np.array(values2) / sum(values2)
        # get embeddings
        keys1 = [key for key, value in tokens1]
        keys2 = [key for key, value in tokens2]
        embeds_text1 = self.word_embeddings(keys1)
        embeds_text2 = self.word_embeddings(keys2)
        # calculate cost matrix
        cost_matrix = np.matmul(embeds_text1, embeds_text2.T)
        cost_matrix = np.ones(cost_matrix.shape) - cost_matrix
        return ot.emd2(dist_text1, dist_text2, cost_matrix)

