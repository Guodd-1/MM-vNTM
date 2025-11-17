"Based on: https://github.com/gonzalezf/multimodal_neural_topic_modeling/"
from collections import OrderedDict
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


# Inference network used by ZeroShotTM: relies solely on contextual embeddings (e.g., SBERT/CLIP text) and image embeddings; **does NOT use BoW**
class ContextualInferenceNetwork(nn.Module):
    """Inference network (encoder) that infers the posterior topic distribution of a document from contextual features (text + image)."""

    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2, label_size=0, img_enc_dim=0, kappa1=10):
        """
        Initialize the inference network.

        Parameters:
        - input_size: BoW vocabulary size (not used here; retained only for interface consistency)
        - bert_size: text embedding dimension (e.g., 512 for CLIP text)
        - output_size: output dimension = number of topics K
        - hidden_sizes: tuple of hidden layer dimensions, e.g., (100, 100)
        - activation: activation function, either 'softplus' or 'relu'
        - dropout: dropout probability
        - label_size: label dimension (for supervised training; 0 means no labels)
        - img_enc_dim: image embedding dimension (e.g., 512 for CLIP image)
        """
        super(ContextualInferenceNetwork, self).__init__()  # Parameter validation
        assert isinstance(input_size, int), "input_size must be of type int."
        assert isinstance(output_size, int), "output_size must be of type int."
        assert isinstance(hidden_sizes, tuple), "hidden_sizes must be of type tuple."
        assert activation in ['softplus', 'relu'], "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.kappa1 = kappa1

        # Set activation function
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        # Here, we receive contextualized embeddings (we can concatenate image embeddings)
        self.input_layer = nn.Linear(bert_size + label_size + img_enc_dim,
                                     hidden_sizes[0])  # Incorporating images into CTM

        # Build intermediate hidden layers
        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(
                nn.Linear(h_in, h_out),
                self.activation
            ))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))
        ]))

        # Replace original f_mu / f_sigma definitions
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_kappa = nn.Linear(hidden_sizes[-1], 1)  # Output: [N, 1]

        # Dropout layer
        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x=None, x_bert=None, labels=None, X_image_embeddings=None):
        """
        Forward pass: infer parameters (μ, κ) of the posterior topic distribution from input features.

        Inputs:
        - x: BoW vector (**ignored in this network**)
        - x_bert: text embeddings [N, D_text]
        - labels: labels (optional) [N, L]
        - X_image_embeddings: image embeddings [N, D_img]

        Returns:
        - mu: posterior mean [N, K]
        - kappa: concentration parameter κ (scalar per document) [N, 1]
        """
        # ZeroShotTM does not use BoW; x_bert is the primary input
        x = x_bert  # x_bert must be provided

        # If labels exist, concatenate them to the text embeddings
        if x_bert is not None and labels is not None:
            x = torch.cat((x_bert, labels), dim=1)

        # If image embeddings exist, concatenate them to the current input
        if X_image_embeddings is not None:
            if x is not None:
                x = torch.cat((x, X_image_embeddings), dim=1)
            else:
                x = X_image_embeddings  # image-only (edge case)

        # Forward pass: input layer → activation → hidden layers → dropout
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)

        # Output posterior parameters
        mu_raw = self.f_mu(x)
        mu = F.normalize(mu_raw, p=2, dim=1)  # L2 normalization → unit vectors

        kappa_raw = self.f_kappa(x)  # [N, 1]
        kappa = F.softplus(
            kappa_raw) + self.kappa1  # softplus ensures >0; self.kappa1 is a preset offset to avoid numerical issues

        return mu, kappa


# Inference network used by CombinedTM: **uses BoW + text embeddings + image embeddings jointly**
class CombinedInferenceNetwork(nn.Module):
    """Inference network (encoder) that fuses BoW, text embeddings, and image embeddings to infer the posterior topic distribution."""

    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2, label_size=0, img_enc_dim=0, kappa1=10):
        """
        Initialize the inference network.

        Parameters:
        - input_size: BoW vocabulary size V
        - bert_size: text embedding dimension D_text
        - output_size: number of topics K
        - other parameters same as above
        """
        super(CombinedInferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must be of type int."
        assert isinstance(output_size, int), "output_size must be of type int."
        assert isinstance(hidden_sizes, tuple), "hidden_sizes must be of type tuple."
        assert activation in ['softplus', 'relu'], "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.kappa1 = kappa1

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        # Key step: project text embeddings into BoW dimension for alignment
        self.adapt_bert = nn.Linear(bert_size, input_size)

        # Input layer: concatenate [BoW + projected text + labels + images] → first hidden layer
        # Total input dim = input_size (BoW) + input_size (adapted BERT) + label_size + img_enc_dim
        self.input_layer = nn.Linear(input_size + input_size + label_size + img_enc_dim, hidden_sizes[0])

        # Hidden layers (same as above)
        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(
                nn.Linear(h_in, h_out),
                self.activation
            ))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))
        ]))

        # Replace original f_mu / f_sigma definitions
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_kappa = nn.Linear(hidden_sizes[-1], 1)  # Output: [N, 1]

        # Dropout layer
        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None, X_image_embeddings=None):
        """
        Forward pass: fuse multimodal inputs to infer topic posterior.

        Inputs:
        - x: BoW vector [N, V]
        - x_bert: text embeddings [N, D_text]
        - labels: labels (optional)
        - X_image_embeddings: image embeddings [N, D_img]

        Returns:
        - mu, kappa: posterior parameters [N, K] and [N, 1]
        """
        # Project text embeddings into BoW space for semantic alignment
        x_bert_adapted = self.adapt_bert(x_bert)  # [N, V]

        # Concatenate BoW and projected text embeddings
        x = torch.cat((x, x_bert_adapted), dim=1)  # [N, 2*V]

        # Concatenate labels (if provided)
        if labels is not None:
            x = torch.cat((x, labels), dim=1)

        # Concatenate image embeddings (if provided)
        if X_image_embeddings is not None:
            x = torch.cat((x, X_image_embeddings), dim=1)

        # Forward propagation
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)

        # Output posterior parameters
        mu_raw = self.f_mu(x)
        mu = F.normalize(mu_raw, p=2, dim=1)  # L2 normalization → unit vectors

        kappa_raw = self.f_kappa(x)  # [N, 1]
        kappa = F.softplus(kappa_raw) + self.kappa1  # Ensure > 0 and avoid numerical instability

        return mu, kappa