"""Constructs data processors and loaders for App sentences,
as well a container object, Args, to hold all arguments passed during
training and evaluation script execution to initialize the BiMPM model.

"""

from abc import ABC
import dill as pickle

import torch
import torch.autograd
from torchtext import data


class DataLoader(ABC):
    """A data loader abstract class, which initializes a field, and prepares
    methods for converting words to chars and building a character vocab.

    """

    def __init__(self, args):
        """Initialize the data loader, store args, and initialize epochs.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.

        """
        self.args = args
        self.TEXT = data.Field(batch_first=True)
        self.last_epoch = -1  # Allow atleast one epoch

    def words_to_chars(self, batch):
        """Convert batch of sentences to appropriately shaped array for
        the WordRepresentationLayer. This will eventually be turned into
        a PyTorch Tensor to track gradients and allow for easy
        backpropagation of errors later on.

        Parameters
        ----------
        batch : Tensor
            A PyTorch Tensor with shape (batch_size, seq_len).

        Returns
        -------
        array_like
            An nested array with shape (batch_size, seq_len, max_word_len).

        """
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return [[self.word_chars[word] for word in sentence]
                for sentence in batch]

    def build_char_vocab(self):
        """Create char vocabulary, generate char2idx and idx2char mapping,
        and pad words to max word length.

        """
        for word in self.TEXT.vocab.itos[2:]:  # Skip <pad> and <unk>
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            # Pad words until max word length
            chars.extend([0] * (self.max_word_len - len(word)))
            self.word_chars.append(chars)

    def keep_training(self, iterator):
        """Track batch iteration and epochs.

        Parameters
        ----------
        iterator : Iterator
            An iterator object which provides batches of data, and keeps a
            track of the epochs.

        Returns
        -------
        bool
            False if all epochs are complete, else True.

        """
        self.present_epoch = int(iterator.epoch)
        if self.present_epoch == self.args.epoch:
            return False
        if self.present_epoch > self.last_epoch:
            print(f'  epoch: {self.present_epoch+1}')
        self.last_epoch = self.present_epoch
        return True


class AppData(DataLoader):
    """A data processor for App data, which inherits from the Quora, and
    DataLoader class and generates a single example and batch for inference.

    """

    def __init__(self, args, app_data=None):
        """Initialize the data loader, create a dataset and a batch.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.
        app_data : list, optional
            A Python list with `q1` and `q2` as keys for two queries
            (default is None).

        """
        super().__init__(args)

        self.TEXT = pickle.load(open('./data/TEXT.pkl', 'rb'))

        self.fields = [('q1', self.TEXT), ('q2', self.TEXT)]

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        # Handle <pad> and <unk>
        self.char_vocab = {'': 0}
        self.word_chars = [[0] * self.max_word_len, [0] * self.max_word_len]

        self.example = [
            data.Example.fromlist(data=app_data, fields=self.fields)
        ]
        self.dataset = data.Dataset(self.example, self.fields)
        self.batch = data.Batch(self.example, self.dataset, device=args.device)

        self.build_char_vocab()


class Sentence:
    """Creates a Sentence object to hold the words and characters for each
    sentence in a batch.

    """

    def __init__(self, batch, model_data, data_type):
        """Initialize a Sentence object for SNLI or Quora data.

        Parameters
        ----------
        batch : Tensor
            A PyTorch Tensor with shape (batch_size, seq_len).
        model_data : {Quora, SNLI}
            A data loading object which returns word vectors and sentences.
        data_type : {'Quora', 'SNLI'}, optional
            Choose either SNLI or Quora (default is 'quora').

        """
        self.batch, self.model_data = batch, model_data

        if data_type.lower() == 'snli':
            self.p, self.q = 'premise', 'hypothesis'
        else:
            self.p, self.q = 'q1', 'q2'

    def process_batch(self, device):
        """Retrieve either SNLI or Quora data from each batch by label, and
        construct words and chars.

        Parameters
        ----------
        device : {'cuda:0', 'cpu'}
            Indicates whether to store the char tensors in the cpu or gpu.

        """
        self.p = getattr(self.batch, self.p)
        self.q = getattr(self.batch, self.q)

        # Track gradients on char tensors
        self.char_p = torch.LongTensor(self.model_data.words_to_chars(self.p))
        self.char_q = torch.LongTensor(self.model_data.words_to_chars(self.q))

        self.char_p = self.char_p.to(device)
        self.char_q = self.char_q.to(device)

    def make_sentence_dict(self):
        """Create a dictionary for words and chars in a sentence."""
        self.p = {'words': self.p, 'chars': self.char_p}
        self.q = {'words': self.q, 'chars': self.char_q}

    def generate(self, device):
        """Generate a sentence dictionary with words and chars for each
        sentence

        Parameters
        ----------
        device : {'cuda:0', 'cpu'}
            Indicates whether to store the char tensors in the cpu or gpu.

        Returns
        -------
        tuple
            A tuple of sentence objects.

        """
        self.process_batch(device)
        self.make_sentence_dict()
        return (self.p, self.q)


class Args:
    """Creates a mapping from dictionary to object."""

    def __init__(self, args_dict):
        """Initialize and store args from dict into self attributes for easy
        access during runtime.

        Parameters
        ---------
        args_dict : dict
            A dictionary of all arguments passed to the training or
            evaluation script.

        """
        for k, v in args_dict.items():
            setattr(self, k, v)
