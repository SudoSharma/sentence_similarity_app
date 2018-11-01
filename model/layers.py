"""Creates the layers for a BiMPM model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterRepresentationEncoder(nn.Module):
    """A character embedding layer with embeddings that are learned along
    with other network parameters during training.

    """

    def __init__(self, args):
        """Initialize the character embedding layer model architecture, and
        the char rnn.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.

        """
        super(CharacterRepresentationEncoder, self).__init__()

        self.char_hidden_size = args.char_hidden_size

        self.char_encoder = nn.Embedding(
            args.char_vocab_size, args.char_input_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=args.char_input_size,
            hidden_size=args.char_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True)

    def forward(self, chars):
        """Defines forward pass computations flowing from inputs to
        outputs in the network.

        Parameters
        ----------
        chars : Tensor
            A PyTorch Tensor with shape (batch_size, seq_len, max_word_len)

        Returns
        -------
        Tensor
            A PyTorch Tensor with shape (batch_size, seq_len,
            char_hidden_size).

        """
        batch_size, seq_len, max_word_len = chars.size()
        chars = chars.view(batch_size * seq_len, max_word_len)

        # out_shape: (1, batch_size * seq_len, char_hidden_size)
        chars = self.lstm(self.char_encoder(chars))[-1][0]

        return chars.view(-1, seq_len, self.char_hidden_size)


class WordRepresentationLayer(nn.Module):
    """A word representation layer which will create word and char embeddings
    which will then be concatenated and trained with other model parameters.

    """

    def __init__(self, args, model_data):
        """Initialize the word representation layer, and store pre-trained
        embeddings. Also initialize the char rnn.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.
        model_data : {Quora, SNLI}
            A data loading object which returns word vectors and sentences.

        """
        super(WordRepresentationLayer, self).__init__()

        self.drop = args.dropout

        self.word_encoder = nn.Embedding(args.word_vocab_size, args.word_dim)
        self.word_encoder.weight.data.copy_(model_data.TEXT.vocab.vectors)
        self.word_encoder.weight.requires_grad = False  # Freeze parameters

        self.char_encoder = CharacterRepresentationEncoder(args)

    def dropout(self, tensor):
        """Defines a dropout function to regularize the parameters.

        Parameters
        ----------
        tensor : Tensor
            A Pytorch Tensor.

        Returns
        -------
        Tensor
            A PyTorch Tensor with same size as input.

        """
        return F.dropout(tensor, p=self.drop, training=self.training)

    def forward(self, p):
        """Defines forward pass computations flowing from inputs to
        outputs in the network.

        Parameters
        ----------
        p : Sentence
            A sentence object with chars and word batches.

        Returns
        -------
        Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            word_dim + char_hidden_size).

        """
        words = self.word_encoder(p['words'])
        chars = self.char_encoder(p['chars'])
        p = torch.cat([words, chars], dim=-1)

        return self.dropout(p)


class ContextRepresentationLayer(nn.Module):
    """A context representation layer to incorporate contextual information
    into the representation of each time step of p and q.

    """

    def __init__(self, args):
        """Initialize the context representation layer, and initialize an
        lstm.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.

        """
        super(ContextRepresentationLayer, self).__init__()

        self.drop = args.dropout
        self.input_size = args.word_dim + args.char_hidden_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True)

    def dropout(self, tensor):
        """Defines a dropout function to regularize the parameters.

        Parameters
        ----------
        tensor : Tensor
            A Pytorch Tensor.

        Returns
        -------
        Tensor
            A PyTorch Tensor with same size as input.

        """
        return F.dropout(tensor, p=self.drop, training=self.training)

    def forward(self, p):
        """Defines forward pass computations flowing from inputs to
        outputs in the network.

        Parameters
        ----------
        p : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            word_dim + char_hidden_size).

        Returns
        -------
        Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes)

        """
        p = self.lstm(p)[0]

        return self.dropout(p)


class MatchingLayer(nn.Module):
    """A matching layer to compare contextual embeddings from one sentence
    against the contextual embeddings of the other sentence.

    """

    def __init__(self, args):
        """Initialize the mactching layer architecture.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.

        """
        super(MatchingLayer, self).__init__()

        self.drop = args.dropout
        self.hidden_size = args.hidden_size
        self.l = args.num_perspectives
        self.W = nn.ParameterList([
            nn.Parameter(torch.rand(self.l, self.hidden_size))
            for _ in range(8)
        ])

    def dropout(self, tensor):
        """Defines a dropout function to regularize the parameters.

        Parameters
        ----------
        tensor : Tensor
            A Pytorch Tensor.

        Returns
        -------
        Tensor
            A PyTorch Tensor with same size as input.

        """
        return F.dropout(tensor, p=self.drop, training=self.training)

    def cat(self, *args):
        """Concatenate matching vectors.

        Parameters
        ----------
        *args
            Variable length argument list.

        Returns
        -------
        Tensor
            A PyTorch Tensor with input tensors concatenated over dim 2.
        """
        return torch.cat(list(args), dim=2)  # dim 2 is num_perspectives 

    def split(self, tensor, direction='fw'):
        """Split the output of an bidirectional rnn into forward or
        backward passes.

        Parameters
        ----------
        tensor : Tensor
            A Pytorch Tensor containing the output of a bidirectional rnn.
        direction : str, optional
            The direction of the rnn pass to return (default is 'fw').

        Returns
        -------
        Tensor
            A Pytorch Tensor for the rnn pass in the specified direction.

        """
        if direction == 'fw':
            return torch.split(tensor, self.hidden_size, dim=-1)[0]
        elif direction == 'bw':
            return torch.split(tensor, self.hidden_size, dim=-1)[-1]

    def match(self,
              p,
              q,
              w,
              direction='fw',
              split=True,
              stack=True,
              cosine=False):
        """Match two sentences based on various matching strategies and
        time-step constraints.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes) if split is True, else it is size
            (batch_size, seq_len, hidden_size).
        w : Parameter
            A Pytorch Parameter with size (num_perspectives, hidden_size).
        direction : str, optional
            The direction of the rnn pass to return (default is 'fw').
        split : bool, optional
            Split input Tensor if output from bidirectional rnn
            (default is True).
        stack : bool, optional
            Stack input Tensor if input size is (batch_size, hidden_size),
            for the second sentence `q` for example, in the case of the
            full-matching strategy, when matching only the last time-step
            (default is True).
        cosine : bool, optional
            Perform cosine similarity using built-in PyTorch Function, for
            example, in the case of a full-matching or attentive-matching
            strategy (default is False).

        Returns
        -------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len, l) with the
            weights multiplied by the input sentence.
        Tensor
            If cosine=True, returns a tensor of size (batch_size, seq_len, l)
            representing the distance between `p` and `q`.

        """
        if split:
            p = self.split(p, direction)
            q = self.split(q, direction)

        if stack:
            seq_len = p.size(1)

            # out_shape: (batch_size, seq_len_p, hidden_size)
            if direction == 'fw':
                q = torch.stack([q[:, -1, :]] * seq_len, dim=1)
            elif direction == 'bw':
                q = torch.stack([q[:, 0, :]] * seq_len, dim=1)

        # out_shape: (1, l, 1, hidden_size)
        w = w.unsqueeze(0).unsqueeze(2)

        # out_shape: (batch_size, l, seq_len_{p, q}, hidden_size)
        p = w * torch.stack([p] * self.l, dim=1)
        q = w * torch.stack([q] * self.l, dim=1)

        if cosine:
            # out_shape: (batch_size, seq_len, l)
            return F.cosine_similarity(p, q, dim=-1).permute(0, 2, 1)

        return (p, q)

    def attention(self, p, q, w, direction='fw', att='mean'):
        """Create either a mean or max attention vector for the attentive
        matching strategies.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes).
        w : Parameter
            A Pytorch Parameter with size (num_perspectives, hidden_size).
        direction : str, optional
            The direction of the rnn pass to return (default is 'fw').
        att : str, optional
            The type of attention vector to generate (default is 'mean').

        Returns
        -------
        att_p_match, att_q_match : Tensor
            A PyTorch Tensor with size (batch_size, seq_len, hidden_size).

        """
        # out_shape: (batch_size, seq_len_{p, q}, hidden_size)
        p = self.split(p, direction)
        q = self.split(q, direction)

        # out_shape: (batch_size, seq_len_p, 1)
        p_norm = p.norm(p=2, dim=2, keepdim=True)
        # out_shape: (batch_size, 1, seq_len_q)
        q_norm = q.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

        # out_shape: (batch_size, seq_len_p, seq_len_q)
        dot = torch.bmm(p, q.permute(0, 2, 1))
        magnitude = p_norm * q_norm
        cosine = dot / magnitude

        # out_shape: (batch_size, seq_len_p, seq_len_q, hidden_size)
        weighted_p = p.unsqueeze(2) * cosine.unsqueeze(-1)
        weighted_q = q.unsqueeze(1) * cosine.unsqueeze(-1)

        if att == 'mean':
            # out_shape: (batch_size, seq_len_{q, p}, hidden_size))
            p_vec = weighted_p.sum(dim=1) /\
                cosine.sum(dim=1, keepdim=True).permute(0, 2, 1)
            q_vec = weighted_q.sum(dim=2) / cosine.sum(dim=2, keepdim=True)
        elif att == 'max':
            # out_shape: (batch_size, seq_len_{q, p}, hidden_size)
            p_vec, _ = weighted_p.max(dim=1)
            q_vec, _ = weighted_q.max(dim=2)

        # out_shape: (batch_size, seq_len_{p, q}, l)
        att_p_match = self.match(
            p, q_vec, w, split=False, stack=False, cosine=True)
        att_q_match = self.match(
            q, p_vec, w, split=False, stack=False, cosine=True)

        return (att_p_match, att_q_match)

    def full_match(self, p, q, w, direction='fw'):
        """Match each contextual embedding with the last time-step of the other
        sentence for either the forward or backward pass.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes).
        w : Parameter
            A Pytorch Parameter with size (num_perspectives, hidden_size).
        direction : str, optional
            The direction of the rnn pass to return (default is 'fw').

        Returns
        -------
        Tensor
            A PyTorch Tensor with size (batch_size, seq_len, l).

        """
        # out_shape: (batch_size, seq_len_{p, q}, l)
        return self.match(
            p, q, w, direction, split=True, stack=True, cosine=True)

    def maxpool_match(self, p, q, w, direction='fw'):
        """Match each contextual embedding with each time-step of the other
        sentence for either the forward or backward pass.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes).
        w : Parameter
            A Pytorch Parameter with size (num_perspectives, hidden_size).
        direction : str, optional
            The direction of the rnn pass to return (default is 'fw').

        Returns
        -------
        pool_p, pool_q : array_like
            A tuple of PyTorch Tensors with size (batch_size, seq_len, l).

        """
        # out_shape: (batch_size, l, seq_len_{p, q}, hidden_size)
        p, q = self.match(
            p, q, w, direction, split=True, stack=False, cosine=False)

        # out_shape: (batch_size, l, seq_len_{p, q}, 1)
        p_norm = p.norm(p=2, dim=-1, keepdim=True)
        q_norm = q.norm(p=2, dim=-1, keepdim=True)

        # out_shape: (batch_size, l, seq_len_p, seq_len_q)
        dot = torch.matmul(p, q.permute(0, 1, 3, 2))
        magnitude = p_norm * q_norm.permute(0, 1, 3, 2)

        # out_shape: (batch_size, seq_len_p, seq_len_q, l)
        cosine = (dot / magnitude).permute(0, 2, 3, 1)

        # out_shape: (batch_size, seq_len_{p, q}, l)
        pool_p, _ = cosine.max(dim=2)
        pool_q, _ = cosine.max(dim=1)

        return (pool_p, pool_q)

    def attentive_match(self, p, q, w, direction='fw'):
        """Match each contextual embedding with its mean attentive vector.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes).
        w : Parameter
            A Pytorch Parameter with size (num_perspectives, hidden_size).
        direction : str, optional
            The direction of the rnn pass to return (default is 'fw').

        Returns
        -------
        array_like
            A tuple of PyTorch Tensors with size (batch_size, seq_len, l).

        """
        # out_shape: (batch_size, seq_len_{p, q}, l)
        return self.attention(p, q, w, direction, att='mean')

    def max_attentive_match(self, p, q, w, direction='fw'):
        """Match each contextual embedding with its max attentive vector.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes).
        w : Parameter
            A Pytorch Parameter with size (num_perspectives, hidden_size).
        direction : str, optional
            The direction of the rnn pass to return (default is 'fw').

        Returns
        -------
        array_like
            A tuple of PyTorch Tensors with size (batch_size, seq_len, l).

        """
        # out_shape: (batch_size, seq_len_{p, q}, l)
        return self.attention(p, q, w, direction, att='max')

    def match_operation(self, p, q, W):
        """Match each contextual embedding with its attentive vector.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes).
        W : ParameterList
            A list of Pytorch Parameters with size
            (num_perspectives, hidden_size).

        Returns
        -------
        array_like
            A list of PyTorch Tensors of size (batch_size, seq_len, l*8).

        """

        full_p2q_fw = self.full_match(p, q, W[0], 'fw')
        full_p2q_bw = self.full_match(p, q, W[1], 'bw')
        full_q2p_fw = self.full_match(q, p, W[0], 'fw')
        full_q2p_bw = self.full_match(q, p, W[1], 'bw')

        pool_p_fw, pool_q_fw = self.maxpool_match(p, q, W[2], 'fw')
        pool_p_bw, pool_q_bw = self.maxpool_match(p, q, W[3], 'bw')

        att_p2mean_fw, att_q2mean_fw = self.attentive_match(p, q, W[4], 'fw')
        att_p2mean_bw, att_q2mean_bw = self.attentive_match(p, q, W[5], 'bw')

        att_p2max_fw, att_q2max_fw = self.max_attentive_match(p, q, W[6], 'fw')
        att_p2max_bw, att_q2max_bw = self.max_attentive_match(p, q, W[7], 'bw')

        # Concatenate all the vectors for each sentence
        p_vec = self.cat(full_p2q_fw, pool_p_fw, att_p2mean_fw, att_p2max_fw,
                         full_p2q_bw, pool_p_bw, att_p2mean_bw, att_p2max_bw)

        q_vec = self.cat(full_q2p_fw, pool_q_fw, att_q2mean_fw, att_q2max_fw,
                         full_q2p_bw, pool_q_bw, att_q2mean_bw, att_q2max_bw)

        # out_shape: (batch_size, seq_len_{p, q}, l*8)
        return (self.dropout(p_vec), self.dropout(q_vec))

    def forward(self, p, q):
        """Defines forward pass computations flowing from inputs to
        outputs in the network.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len,
            hidden_size, num_passes).

        Returns
        -------
        array_like
            A list of PyTorch Tensors of size (batch_size, seq_len, l*8).

        """
        return self.match_operation(p, q, self.W)


class AggregationLayer(nn.Module):
    """An aggregation layer to combine two sequences of matching vectors into
    fixed-length matching vector.

    """

    def __init__(self, args):
        """Initialize the aggregation layer architecture.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.

        """
        super(AggregationLayer, self).__init__()

        self.hidden_size = args.hidden_size
        self.drop = args.dropout
        self.lstm = nn.LSTM(
            input_size=args.num_perspectives * 8,
            hidden_size=args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True)

    def dropout(self, tensor):
        """Defines a dropout function to regularize the parameters.

        Parameters
        ----------
        tensor : Tensor
            A Pytorch Tensor.

        Returns
        -------
        Tensor
            A PyTorch Tensor with same size as input.

        """
        return F.dropout(tensor, p=self.drop, training=self.training)

    def forward(self, p, q):
        """Defines forward pass computations flowing from inputs to
        outputs in the network.

        Parameters
        ----------
        p, q : Tensor
            A PyTorch Tensor with size (batch_size, seq_len, l*8).

        Returns
        -------
        Tensor
            A PyTorch Tensor of size (batch_size, hidden_size*4).

        """
        # out_shape: (2, batch_size, hidden_size)
        p = self.lstm(p)[-1][0]
        q = self.lstm(q)[-1][0]

        # out_shape: (batch_size, hidden_size*4)
        x = torch.cat([
            p.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2),
            q.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)
        ],
                      dim=1)

        return self.dropout(x)


class PredictionLayer(nn.Module):
    """An prediction layer to evaluate the probability distribution for a class
    given the two sentences. The number of outputs would change based on task.

    """

    def __init__(self, args):
        """Initialize the prediction layer architecture.

        Parameters
        ----------
        args : Args
            An object with all arguments for BiMPM model.

        """
        super(PredictionLayer, self).__init__()

        self.drop = args.dropout
        self.hidden_layer = nn.Linear(args.hidden_size * 4,
                                      args.hidden_size * 2)
        self.output_layer = nn.Linear(args.hidden_size * 2, args.class_size)

    def dropout(self, tensor):
        """Defines a dropout function to regularize the parameters.

        Parameters
        ----------
        tensor : Tensor
            A Pytorch Tensor.

        Returns
        -------
        Tensor
            A PyTorch Tensor with same size as input.

        """
        return F.dropout(tensor, p=self.drop, training=self.training)

    def forward(self, match_vec):
        """Defines forward pass computations flowing from inputs to
        outputs in the network.

        Parameters
        ----------
        match_vec : Tensor
            A PyTorch Tensor of size (batch_size, hidden_size*4).

        Returns
        -------
        Tensor
            A PyTorch Tensor of size (batch_size, class_size).

        """
        x = F.relu(self.hidden_layer(match_vec))

        return self.output_layer(self.dropout(x))
