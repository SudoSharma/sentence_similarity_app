"""Tests a model trained on a PyTorch reimplementation of BiMPM"""

import torch

from model.bimpm import BiMPM
from model.utils import AppData, Sentence


def evaluate(model, args, model_data):
    """Test the BiMPM model on App data.

    Parameters
    ----------
    args : Args
        An object with all arguments for BiMPM model.
    model_data : AppData
        A data loading object which returns word vectors and sentences.

    Returns
    -------
    preds : Tensor
        A length-2 PyTorch tensor of predictions for similar or asimilar class.

    """
    cdef object preds, p, q
    model.eval()

    p, q = Sentence(model_data.batch, model_data,
                    args.data_type).generate(args.device)
    preds = model(p, q)
    return preds.data[0]


def load_model(args, model_data):
    """Load the trained BiMPM model for testing

    Parameters
    ----------
    args : Args
        An object with all arguments for BiMPM model
    model_data : AppData
        A data loading object which returns word vectors and sentences.

    Returns
    -------
    model : BiMPM
        A new model initialized with the weights from the provided trained
        model.
    """
    cdef object state_dict
    model = BiMPM(args, model_data)
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)

    return model
