from gensim.parsing.preprocessing import preprocess_string
import math

# used for documentation
from typing import List


class Vocabulary:
    """The class for storing vocabulary

    The vocabulary class is used to index the words of a particular
    dataset and for storing some basic word statistics.

    Attributes:
        word2index: The word-to-index dictionary.
        index2word: The index-to-word dictionary.
        word_count: The dictionary containing the word's frequency.
        n_words: The number of unique words in the vocabulary.
        word_document_count: The dictionary containing the word's document count.
        n_documents: The number of documents.
    """

    def __init__(self, padding: bool = False) -> None:
        """Initializes the vocabulary instance

        Args:
            padding: The boolean value reserved for adding
                a padding component to the vocabulary; 0
                is reserved for filling the context words
                (Default: False).
        """
        self.word2index = {}
        self.index2word = {}
        self.word_count = {}
        self.n_words = 0
        self.word_doc_count = {}
        self.n_documents = 0
        self.padding = padding

    def add_word(self, word: str) -> None:
        """Add a word to the vocabulary.

        Args:
            word: The added word.
        """
        if word in self.word_count:
            self.word_count[word] += 1
        else:
            index = self.n_words + (1 if self.padding else 0)
            self.word2index[word] = index
            self.index2word[index] = word
            self.word_count[word] = 1
            self.n_words += 1

    def add_word_document(self, word: str) -> None:
        """Update the word-to-document count statistics.

        Args:
            word: The word to update statistics.
        """
        if word in self.word_doc_count:
            self.word_doc_count[word] += 1
        else:
            self.word_doc_count[word] = 1

    def add_document(self, document: str) -> None:
        """Adds the document's terms to the vocabulary

        Args:
            document: The added document.
        """
        # iterate through the document terms
        self.n_documents += 1
        document_terms = set()  # store the unique document terms
        terms = preprocess_string(document, [])
        for term in terms:
            # update the term statistics
            self.add_word(term)
            document_terms.add(term)
        # update the word-to-document statistics
        for term in document_terms:
            self.add_word_document(term)

    def add_corpus(self, corpus: List[str]) -> None:
        """Adds the corpus of documents to the vocabulary

        Args:
            corpus: The list of documents.
        """
        for document in corpus:
            self.add_document(document)

    def get_idf(self, word: str) -> float:
        """Gets the idf value of the given word

        Calculates the inverse document frequency (IDF)
        of the given word. It is calculated as `log(N/n)`,
        where `N` is the total number of documents the
        vocabulary has seen and `n` is the number of
        documents containing the word.

        Args:
            word: The word string.

        Returns:
            The float number representing the idf score
            of the term.
        """
        word_doc_freq = self.word_doc_count[word] if word in self.word_doc_count else 0
        return math.log(self.n_documents / (1 + word_doc_freq))

    def get_ft(self, word: str) -> float:
        """Gets the ft value of the given word

        Retrieves the term frequency (TF) value of the
        given word.

        Args:
            word: The word string.

        Returns:
            The float number representing the ft score
            of the term.
        """
        return self.word_count[word] if word in self.word_count else 0
