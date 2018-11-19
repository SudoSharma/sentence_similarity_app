"""Creates a BiMPM model architecture based on PyTorch's nn.Module."""

import torch.nn as nn
import model.layers as L


class BiMPM(nn.Module):
    """A bilateral multi-perspective matching model used for a variety of
    NLP tasks, including paraphrase identification, natural language
    inference,and asnwer selection.

    """

    def __init__(self, args, model_data):
        """Initialize the BiMPM model architecture.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.
        model_data : {Quora, SNLI}
            A data loading object which returns word vectors and sentences.

        """
        super(BiMPM, self).__init__()

        self.args = args

        cdef object w_layer
        cdef object c_layer
        cdef object m_layer
        cdef object a_layer
        cdef object p_layer
        self.w_layer = L.WordRepresentationLayer(args, model_data)
        self.c_layer = L.ContextRepresentationLayer(args)
        self.m_layer = L.MatchingLayer(args)
        self.a_layer = L.AggregationLayer(args)
        self.p_layer = L.PredictionLayer(args)

    def forward(self, p, q):
        """Defines forward pass computations flowing from inputs to
        outputs in the network.

        Parameters
        ----------
        p, q : Sentence
            A sentence object with chars and word batches.

        Returns
        -------
        torch.FloatTensor
            A PyTorch FloatTensor object with size (batch_size, num_classes)
            containing the class probalities for each sentence in the batch.

        """
        cdef object match_vec
        p, q = self.w_layer(p), self.w_layer(q)  # Create word embeddings
        p, q = self.c_layer(p), self.c_layer(q)  # Incorporate context
        p, q = self.m_layer(p, q)  # Compare contexts
        match_vec = self.a_layer(p, q)  # Aggregate context vectors

        return self.p_layer(match_vec)  # Generate task-specific probabilities
