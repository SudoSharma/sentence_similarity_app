"""Tests a model trained on a PyTorch reimplementation of BiMPM"""

import os
import csv
import plac
import dill as pickle

import torch
from torch import nn

from model.bimpm import BiMPM
from model.utils import AppData, Sentence, Args


def main(shutdown: ("shutdown system after training", 'flag', 's'),
         research: ("use medium dataset", 'flag', 'r'),
         travis: ("use small testing dataset", 'flag', 't'),
         app: ("evaluate user queries from app", 'flag', 'a'),
         model_path,
         batch_size: (None, 'option', None, int) = 64,
         char_input_size: (None, 'option', None, int) = 20,
         char_hidden_size: (None, 'option', None, int) = 50,
         data_type: ("use quora, snli, or app", 'option', None, str,
                     ['quora', 'snli', 'app']) = 'quora',
         dropout: (None, 'option', None, float) = 0.1,
         epoch: (None, 'option', None, int) = 10,
         hidden_size: (None, 'option', None, int) = 100,
         lr: (None, 'option', None, float) = 0.001,
         num_perspectives: (None, 'option', None, int) = 20,
         print_interval: (None, 'option', None, int) = 500,
         word_dim: (None, 'option', None, int) = 300):
    """Print the best BiMPM model accuracy for the test set in a cycle.

    Parameters
    ----------
    shutdown : bool, flag
        Shutdown system after training (default is False).
    research : bool, flag
        Run experiments on medium dataset (default is False).
    travis : bool, flag
        Run tests on small dataset (default is False)
    app : bool, flag
        Whether to evaluate queries from bimpm app (default is False).
    model_path : str
        A path to the location of the BiMPM trained model.
    batch_size : int, optional
        Number of examples in one iteration (default is 64).
    char_input_size : int, optional
        Size of character embedding (default is 20).
    char_hidden_size : int, optional
        Size of hidden layer in char lstm (default is 50).
    data_type : {'Quora', 'SNLI'}, optional
        Choose either SNLI or Quora (default is 'quora').
    dropout : int, optional
        Applied to each layer (default is 0.1).
    epoch : int, optional
        Number of passes through full dataset (default is 10).
    hidden_size : int, optional
        Size of hidden layer for all BiLSTM layers (default is 100).
    lr : int, optional
        Learning rate (default is 0.001).
    num_perspectives : int, optional
        Number of perspectives in matching layer (default is 20).
    word_dim : int, optional
        Size of word embeddings (default is 300).

    Raises
    ------
    RuntimeError
        If any data source other than SNLI or Quora is requested.

    """
    # Store local namespace dict in Args() object
    args = Args(locals())

    args.device = torch.device('cuda:0' if torch.cuda.
                               is_available() else 'cpu')

    # Hanlde research and travis mode
    if (args.research or args.travis) and args.data_type.lower() == 'snli':
        raise RuntimeError("Invalid dataset size specified for SNLI data.")

    if args.research:
        print('Research mode detected. Lowering print interval...')
        args.print_interval = 5
    if args.travis:
        print('Travis mode detected. Adjusting parameters...')
        args.epoch = 2
        args.batch_size = 2
        args.print_interval = 1

    if app:
        # Load sample queries and model_data for app mode
        help_message = ("\nPlease create a csv file "
                        "`./app_data/sample_queries.csv` with two queries."
                        " For example:"
                        "\n\t$ cat sample_queries.csv"
                        "\n\tHow can I run faster?"
                        "\n\tHow do I get better at running?\n")

        try:
            with open('./app_data/sample_queries.csv', 'r') as f:
                reader = csv.reader(f)
                app_data = []
                [app_data.extend(line) for line in reader]
            assert len(
                app_data) == 2, f"Too many queries to unpack. {help_message}"
        except FileNotFoundError as e:
            print(e)
            print(help_message)
            return

        print("Loading App data...")
        model_data = AppData(args, app_data)
    elif args.data_type.lower() == 'snli':
        print("Loading SNLI data...")
        model_data = SNLI(args)
    elif args.data_type.lower() == 'quora':
        print("Loading Quora data...")
        model_data = Quora(args)
    else:
        raise RuntimeError(
            'Data source other than SNLI or Quora was provided.')

    # Create a few more parameters based on chosen dataset
    args.word_vocab_size = len(model_data.TEXT.vocab)
    args.char_vocab_size = len(model_data.char_vocab)
    args.class_size = len(model_data.LABEL.vocab)
    args.max_word_len = model_data.max_word_len

    print("Loading model...")
    model = load_model(args, model_data)

    if app:
        # Store args for use in app
        pickle_dir = './app_data/'
        args_pickle = 'args.pkl'
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        pickle.dump(args, open(f'{pickle_dir}{args_pickle}', 'wb'))

        preds = evaluate(model, args, model_data, mode='app')

        print('\nQueries:\n', f'\n{app_data[0]}\n', f'{app_data[1]}\n', sep='')
        print('\nPrediction:')
        if max(preds) == preds.data[1]:
            print('\nSIMILAR based on max value at index 1:',
                  f'\npreds:  {preds.data}\n')
        else:
            print('\nNOT SIMILAR based on max value at index 0',
                  f'\npreds:  {preds.data}\n')
    else:
        _, eval_acc = evaluate(model, args, model_data, mode='eval')
        print(f'\neval_acc:  {eval_acc:.3f}\n')


def evaluate(model, args, model_data, mode='eval'):
    """Test the BiMPM model on SNLI or Quora validation or test data.

    Parameters
    ----------
    args : Args
        An object with all arguments for BiMPM model.
    model_data : {Quora, SNLI}
        A data loading object which returns word vectors and sentences.
    mode : int, optional
        Indicates whether to use `valid`, `eval`, or `app` data
        (default is 'eval').

    Returns
    -------
    loss : int
        The loss of the model provided.
    acc : int
        The accuracy of the model provided.
    preds : Tensor
        A length-2 PyTorch tensor of predictions for similar or asimilar class.

    """
    model.eval()

    if mode == 'valid':
        iterator = model_data.valid_iter
    elif mode == 'eval':
        iterator = model_data.eval_iter
    elif mode == 'app':
        p, q = Sentence(model_data.batch, model_data,
                        args.data_type).generate(args.device)
        preds = model(p, q)
        return preds.data[0]

    criterion = nn.CrossEntropyLoss()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        # Sentence() object contains chars and word batches
        p, q = Sentence(batch, model_data,
                        args.data_type).generate(args.device)

        preds = model(p, q)
        batch_loss = criterion(preds, batch.label)
        loss += batch_loss.data.item()

        # Retrieve index of class with highest score and calculate accuracy
        _, preds = preds.max(dim=1)
        acc += (preds == batch.label).sum().float()
        size += len(preds)

    acc /= size
    acc = acc.cpu().data.item()
    return loss, acc


def load_model(args, model_data):
    """Load the trained BiMPM model for testing

    Parameters
    ----------
    args : Args
        An object with all arguments for BiMPM model
    model_data : {Quora, SNLI}
        A data loading object which returns word vectors and sentences.

    Returns
    -------
    model : BiMPM
        A new model initialized with the weights from the provided trained
        model.
    """
    model = BiMPM(args, model_data)
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)

    return model


if __name__ == '__main__':
    plac.call(main)  # Only executed when script is run directly
