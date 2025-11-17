"Based on: https://github.com/gonzalezf/multimodal_neural_topic_modeling/"
import torch
from prompt_toolkit.key_binding.bindings.named_commands import self_insert
from torch import nn
from torch.nn import functional as F
import sys
import os
import math
from collections import OrderedDict

sys.path.insert(0, os.path.abspath('../../'))
from .inference_network import CombinedInferenceNetwork, ContextualInferenceNetwork
from ..utils.hyperspherical_vae.distributions.hyperspherical_uniform import HypersphericalUniform
from ..utils.hyperspherical_vae.distributions.von_mises_fisher import VonMisesFisher
from ..utils.hyperspherical_vae.ops.ive import ive

class DecoderNetworkMultimodal(nn.Module):

    def __init__(self, input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0, img_enc_dim = 0, kappa1 = 10, kappa2 = 10, temp = 10, embedding_matrix = None):

        super(DecoderNetworkMultimodal, self).__init__()

        # Parameter validation
        assert isinstance(input_size, int), "input_size must be of type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be of type int and > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model_type must be 'prodLDA' or 'LDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be of type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        # Store hyperparameters
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None
        self.img_enc_dim = img_enc_dim
        self.softmax = nn.Softmax(dim=-1)
        self.kappa1 = kappa1
        self.temp = temp

        # ========== Initialize inference network (encoder) ==========
        if infnet == "zeroshot":
            print('ZeroShotTM THIS IS bert_size', bert_size)
            print('ZeroShotTM THIS IS input_size', input_size)
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size, img_enc_dim=img_enc_dim, kappa1 = self.kappa1)
        elif infnet == "combined":
            print('CombinedTM THIS IS bert_size', bert_size)
            print('CombinedTM THIS IS input_size', input_size)
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size,  img_enc_dim=img_enc_dim, kappa1 = self.kappa1)
        else:
            raise Exception('Missing infnet parameter; options are "zeroshot" and "combined".')

        # ========== Label classification head (optional) ==========
        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        # ========== Initialize reconstruction matrices (decoder core) ==========
        # Topic → Word (for BoW reconstruction)
        self.beta = torch.Tensor(n_components, input_size)
        # Topic → Image features (for image embedding reconstruction)
        self.beta_img = torch.Tensor(n_components, img_enc_dim)
        # Topic → Text features (for text embedding reconstruction)
        self.beta_text_features = torch.Tensor(n_components, bert_size)

        # Move to GPU
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
            self.beta_img = self.beta_img.cuda()
            self.beta_text_features = self.beta_text_features.cuda()

        # Register as learnable parameters
        self.beta = nn.Parameter(self.beta)
        self.beta_img = nn.Parameter(self.beta_img)
        self.beta_text_features = nn.Parameter(self.beta_text_features)

        # Xavier uniform initialization (for variance stability)
        nn.init.xavier_uniform_(self.beta)
        nn.init.xavier_uniform_(self.beta_img)
        nn.init.xavier_uniform_(self.beta_text_features)

        # BatchNorm for word distribution normalization in prodLDA (no affine transform)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # Dropout layer (applied to topic distribution θ)
        self.drop_theta = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None, X_image_embeddings=None,
                txt_reconstruction=True, img_reconstruction=True):


        # ========== Encoding stage: infer posterior distribution ==========
        mu, kappa = self.inf_net(x, x_bert, labels, X_image_embeddings)  # [N, K], [N, 1]
        q_z = VonMisesFisher(mu, kappa)  # posterior
        p_z = HypersphericalUniform(self.n_components - 1, device=mu.device)  # prior

        z = q_z.rsample()  # [N, K], already on unit hypersphere — no need for softmax?
        theta = self.softmax(self.temp * z)

        kl_div = torch.distributions.kl.kl_divergence(q_z, p_z).mean().cuda()  # scalar

        # ========== Decoding stage: reconstruct multimodal data ==========
        if self.model_type == 'prodLDA':
            # BoW reconstruction: θ @ beta → BatchNorm → Softmax
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1
            )

            # Image embedding reconstruction (optional)
            if img_reconstruction:
                img_feature_dists = torch.matmul(z, self.beta_img)
            else:
                img_feature_dists = None
            # Text embedding reconstruction (optional)
            if txt_reconstruction:
                predicted_textual_features = torch.matmul(z, self.beta_text_features)
            else:
                predicted_textual_features = None

            # Save decoding matrices (for inference)
            self.topic_word_matrix = self.beta
            if img_reconstruction:
                self.topic_img_feature_matrix = self.beta_img
            if txt_reconstruction:
                self.topic_text_feature_matrix = self.beta_text_features

        elif self.model_type == 'LDA':

            # LDA mode: apply softmax to beta for simplex constraint
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            self.topic_word_matrix = beta
            word_dist = torch.matmul(theta, beta)

            # Image embedding reconstruction (optional)
            if img_reconstruction:
                img_feature_dists = torch.matmul(theta, self.beta_img)
            else:
                img_feature_dists = None

            # Text embedding reconstruction (optional)
            if txt_reconstruction:
                predicted_textual_features = torch.matmul(theta, self.beta_text_features)
            else:
                predicted_textual_features = None

            if img_reconstruction:
                self.topic_img_feature_matrix = self.beta_img
            if txt_reconstruction:
                self.topic_text_feature_matrix = self.beta_text_features

        else:
            raise NotImplementedError("Only 'prodLDA' and 'LDA' are supported.")

        # ========== Label classification (optional) ==========
        estimated_labels = None
        if labels is not None:
            estimated_labels = self.label_classification(theta)

        # ========== Return all necessary outputs ==========
        return kl_div, word_dist, estimated_labels, img_feature_dists, predicted_textual_features

    def get_theta(self, x=None, x_bert=None, labels=None, X_image_embeddings=None):
        """
        Get document-topic distribution θ at inference time (no gradients; for evaluation/visualization).
        """
        with torch.no_grad():
            mu, kappa = self.inf_net(x, x_bert, labels, X_image_embeddings)
            q_z = VonMisesFisher(mu, kappa)
            z = q_z.rsample()  # [N, K], already on unit hypersphere — no need for softmax?
            theta = self.softmax(self.temp * z)
            return z, theta

class DecoderNetworkMultimodal_ETM(nn.Module):

    def __init__(self, input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0, img_enc_dim = 0, kappa1 = 10, kappa2 = 10, temp = 10, embedding_matrix = None):
        super(DecoderNetworkMultimodal_ETM, self).__init__()

        # Parameter validation
        assert isinstance(input_size, int), "input_size must be of type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be of type int and > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model_type must be 'prodLDA' or 'LDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be of type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        # Store hyperparameters
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None
        self.img_enc_dim = img_enc_dim
        self.softmax = nn.Softmax(dim=-1)
        self.kappa1 = kappa1
        self.temp = temp

        # ========== Initialize inference network (encoder) ==========
        if infnet == "zeroshot":
            print('ZeroShotTM THIS IS bert_size', bert_size)
            print('ZeroShotTM THIS IS input_size', input_size)
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size, img_enc_dim=img_enc_dim, kappa1 = self.kappa1)
        elif infnet == "combined":
            print('CombinedTM THIS IS bert_size', bert_size)
            print('CombinedTM THIS IS input_size', input_size)
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size,  img_enc_dim=img_enc_dim, kappa1 = self.kappa1)
        else:
            raise Exception('Missing infnet parameter; options are "zeroshot" and "combined".')

        # ========== Label classification head (optional) ==========
        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        # ========== Initialize reconstruction matrices (decoder core) ==========
        # Topic → Word (for BoW reconstruction)
        self.beta = None
        self.word_embedding = torch.Tensor(input_size, 100)
        self.topic_embedding = torch.Tensor(n_components, 100)
        # Topic → Image features (for image embedding reconstruction)
        self.beta_img = torch.Tensor(n_components, img_enc_dim)
        # Topic → Text features (for text embedding reconstruction)
        self.beta_text_features = torch.Tensor(n_components, bert_size)

        # Move to GPU
        if torch.cuda.is_available():
            self.word_embedding = self.word_embedding.cuda()
            self.topic_embedding = self.topic_embedding.cuda()
            self.beta_img = self.beta_img.cuda()
            self.beta_text_features = self.beta_text_features.cuda()

        # Register as learnable parameters
        self.word_embedding = nn.Parameter(self.word_embedding)
        self.topic_embedding = nn.Parameter(self.topic_embedding)
        self.beta_img = nn.Parameter(self.beta_img)
        self.beta_text_features = nn.Parameter(self.beta_text_features)

        # Xavier uniform initialization (for variance stability)
        nn.init.xavier_uniform_(self.word_embedding)
        nn.init.xavier_uniform_(self.topic_embedding)
        nn.init.xavier_uniform_(self.beta_img)
        nn.init.xavier_uniform_(self.beta_text_features)

        # BatchNorm for word distribution normalization in prodLDA (no affine transform)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # Dropout layer (applied to topic distribution θ)
        self.drop_theta = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None, X_image_embeddings=None,
                txt_reconstruction=True, img_reconstruction=True):

        # ========== Encoding stage: infer posterior distribution ==========
        mu, kappa = self.inf_net(x, x_bert, labels, X_image_embeddings)  # [N, K], [N, 1]
        q_z = VonMisesFisher(mu, kappa)  # posterior
        p_z = HypersphericalUniform(self.n_components - 1, device=mu.device)  # prior

        z = q_z.rsample()  # [N, K], already on unit hypersphere — no need for softmax?
        theta = self.softmax(self.temp * z)

        kl_div = torch.distributions.kl.kl_divergence(q_z, p_z).mean().cuda()  # scalar

        # ========== Decoding stage: reconstruct multimodal data ==========
        if self.model_type == 'prodLDA':
            # BoW reconstruction: θ @ beta → BatchNorm → Softmax
            beta = self.topic_embedding @ self.word_embedding.transpose(0, 1)
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, beta)), dim=1
            )

            # Image embedding reconstruction (optional)
            if img_reconstruction:
                img_feature_dists = torch.matmul(z, self.beta_img)
            else:
                img_feature_dists = None
            # Text embedding reconstruction (optional)
            if txt_reconstruction:
                predicted_textual_features = torch.matmul(z, self.beta_text_features)
            else:
                predicted_textual_features = None

            # Save decoding matrices (for inference)
            self.topic_word_matrix = beta
            self.beta = beta
            if img_reconstruction:
                self.topic_img_feature_matrix = self.beta_img
            if txt_reconstruction:
                self.topic_text_feature_matrix = self.beta_text_features

        elif self.model_type == 'LDA':

            # LDA mode: apply softmax to beta for simplex constraint
            beta = self.topic_embedding @ self.word_embedding.transpose(0, 1)
            beta = F.softmax(self.beta_batchnorm(beta), dim=1)
            self.topic_word_matrix = beta
            self.beta = beta
            word_dist = torch.matmul(theta, beta)

            # Image embedding reconstruction (optional)
            if img_reconstruction:
                img_feature_dists = torch.matmul(theta, self.beta_img)
            else:
                img_feature_dists = None

            # Text embedding reconstruction (optional)
            if txt_reconstruction:
                predicted_textual_features = torch.matmul(theta, self.beta_text_features)
            else:
                predicted_textual_features = None

            if img_reconstruction:
                self.topic_img_feature_matrix = self.beta_img
            if txt_reconstruction:
                self.topic_text_feature_matrix = self.beta_text_features

        else:
            raise NotImplementedError("Only 'prodLDA' and 'LDA' are supported.")

        # ========== Label classification (optional) ==========
        estimated_labels = None
        if labels is not None:
            estimated_labels = self.label_classification(theta)

        # ========== Return all necessary outputs ==========
        return kl_div, word_dist, estimated_labels, img_feature_dists, predicted_textual_features

    def get_theta(self, x=None, x_bert=None, labels=None, X_image_embeddings=None):
        """
        Get document-topic distribution θ at inference time (no gradients; for evaluation/visualization).
        """
        with torch.no_grad():
            mu, kappa = self.inf_net(x, x_bert, labels, X_image_embeddings)
            q_z = VonMisesFisher(mu, kappa)
            z = q_z.rsample()  # [N, K], already on unit hypersphere — no need for softmax?
            theta = self.softmax(self.temp * z)
            return z, theta

class DecoderNetworkMultimodal_vMFMix(nn.Module):


    def __init__(self, input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0, img_enc_dim = 0, kappa1 = 10, kappa2 = 10, temp = 10, embedding_matrix=None):
        super(DecoderNetworkMultimodal_vMFMix, self).__init__()

        # Parameter validation
        assert isinstance(input_size, int), "input_size must be of type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be of type int and > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model_type must be 'prodLDA' or 'LDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be of type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        # Store hyperparameters
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None
        self.beta = None
        self.img_enc_dim = img_enc_dim
        self.softmax = nn.Softmax(dim=-1)
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.temp = temp
        self.embedding_matrix = embedding_matrix

        # ========== Initialize inference network (encoder) ==========
        if infnet == "zeroshot":
            print('ZeroShotTM THIS IS bert_size', bert_size)
            print('ZeroShotTM THIS IS input_size', input_size)
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size, img_enc_dim=img_enc_dim, kappa1 = self.kappa1)
        elif infnet == "combined":
            print('CombinedTM THIS IS bert_size', bert_size)
            print('CombinedTM THIS IS input_size', input_size)
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size,  img_enc_dim=img_enc_dim, kappa1 = self.kappa1)
        else:
            raise Exception('Missing infnet parameter; options are "zeroshot" and "combined".')

        # ========== Label classification head (optional) ==========
        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        # ========== Initialize reconstruction matrices (decoder core) ==========
        # Word embedding — randomly initialized
        self.word_embedding = torch.Tensor(input_size, 300)
        self.word_embedding = self.word_embedding.cuda()
        self.word_embedding = nn.Parameter(self.word_embedding)
        nn.init.xavier_uniform_(self.word_embedding)

        # Word embedding — initialized from Word2Vec (commented out)
        # embedding_tensor = torch.from_numpy(self.embedding_matrix).float().cuda()
        # embedding_tensor = F.normalize(embedding_tensor, p=2, dim=1)
        # self.word_embedding = nn.Parameter(embedding_tensor)

        # Create topic mean vectors (μ) and concentration parameters (κ), and register as learnable parameters
        self.mu_mat = torch.Tensor(n_components, 300)  # topic mean directions
        self.mu_mat = self.mu_mat.cuda()
        self.mu_mat = nn.Parameter(self.mu_mat)
        nn.init.xavier_uniform_(self.mu_mat)

        self.kappa = torch.Tensor(n_components)  # topic concentration parameters
        self.kappa = self.kappa.cuda()
        self.kappa = nn.Parameter(self.kappa)
        self.kappa.data.fill_(self.kappa2)  # initialize to a constant value

        # Topic → Image features (for image embedding reconstruction)
        self.beta_img = torch.Tensor(n_components, img_enc_dim)
        self.beta_img = self.beta_img.cuda()
        self.beta_img = nn.Parameter(self.beta_img)
        nn.init.xavier_uniform_(self.beta_img)

        # Topic → Text features (for text embedding reconstruction)
        self.beta_text_features = torch.Tensor(n_components, bert_size)
        self.beta_text_features = self.beta_text_features.cuda()
        self.beta_text_features = nn.Parameter(self.beta_text_features)
        nn.init.xavier_uniform_(self.beta_text_features)

        # BatchNorm for word distribution normalization in prodLDA (no affine transform)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # Dropout layer (applied to topic distribution θ)
        self.drop_theta = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None, X_image_embeddings=None,
                txt_reconstruction=True, img_reconstruction=True):

        # print("kappa2 init val:", self.kappa2)
        # print("self.kappa.data before:", self.kappa.data[:5])  # inspect first 5 κs

        # ========== Encoding stage: infer posterior distribution ==========
        mu, kappa = self.inf_net(x, x_bert, labels, X_image_embeddings)  # [N, K], [N, 1]

        # Inspect three statistics:
        # print(">>> kappa1 =", self.kappa1)
        # print(">>> kappa (posterior)  min/mean/max:",
        #       kappa.min().item(), kappa.mean().item(), kappa.max().item())
        # print(">>> kappa std (across docs):", kappa.std().item())  # ← key diagnostic!
        # print(">>> delta_kappa = kappa - kappa1  std:", (kappa - self.kappa1).std().item())

        q_z = VonMisesFisher(mu, kappa)  # posterior
        p_z = HypersphericalUniform(self.n_components - 1, device=mu.device)  # prior

        z = q_z.rsample()  # [N, K],
        theta = self.softmax(self.temp * z)

        kl_div = torch.distributions.kl.kl_divergence(q_z, p_z).mean().cuda()  # scalar

        # ========== Decoding stage: reconstruct multimodal data ==========
        if self.model_type == 'prodLDA':
            # BoW reconstruction: θ @ beta → BatchNorm → Softmax
            beta = self.log_vmf_pdf(self.mu_mat, self.kappa, self.word_embedding)
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, beta)), dim=1
            )

            # Image embedding reconstruction (optional)
            if img_reconstruction:
                img_feature_dists = torch.matmul(z, self.beta_img)
            else:
                img_feature_dists = None
            # Text embedding reconstruction (optional)
            if txt_reconstruction:
                predicted_textual_features = torch.matmul(z, self.beta_text_features)
            else:
                predicted_textual_features = None

            # Save decoding matrices (for inference)
            self.topic_word_matrix = beta
            self.beta = beta
            if img_reconstruction:
                self.topic_img_feature_matrix = self.beta_img
            if txt_reconstruction:
                self.topic_text_feature_matrix = self.beta_text_features

        elif self.model_type == 'LDA':

            # LDA mode: apply softmax to beta for simplex constraint
            beta = self.log_vmf_pdf(self.mu_mat, self.kappa, self.word_embedding)
            beta = F.softmax(self.beta_batchnorm(beta), dim=1)
            # beta = F.softmax(beta)
            self.topic_word_matrix = beta
            self.beta = beta
            word_dist = torch.matmul(theta, beta)

            # Image embedding reconstruction (optional)
            if img_reconstruction:
                img_feature_dists = torch.matmul(theta, self.beta_img)
            else:
                img_feature_dists = None

            # Text embedding reconstruction (optional)
            if txt_reconstruction:
                predicted_textual_features = torch.matmul(theta, self.beta_text_features)
            else:
                predicted_textual_features = None

            if img_reconstruction:
                self.topic_img_feature_matrix = self.beta_img
            if txt_reconstruction:
                self.topic_text_feature_matrix = self.beta_text_features

        else:
            raise NotImplementedError("Only 'prodLDA' and 'LDA' are supported.")

        # ========== Label classification (optional) ==========
        estimated_labels = None
        if labels is not None:
            estimated_labels = self.label_classification(theta)

        # ========== Return all necessary outputs ==========
        return kl_div, word_dist, estimated_labels, img_feature_dists, predicted_textual_features

    def get_theta(self, x=None, x_bert=None, labels=None, X_image_embeddings=None):
        """
        Get document-topic distribution θ at inference time (no gradients; for evaluation/visualization).
        """
        with torch.no_grad():
            mu, kappa = self.inf_net(x, x_bert, labels, X_image_embeddings)
            q_z = VonMisesFisher(mu, kappa)
            z = q_z.rsample()  # [N, K], already on unit hypersphere — no need for softmax?
            theta = self.softmax(self.temp * z)
            return z, theta

    def log_vmf_pdf(self, mu, kappa, vec):
        """
        Compute log-probability of words under a mixture of von Mises-Fisher distributions.
        Each topic corresponds to a vMF: p(w | topic i) ∝ exp(κ_i * ⟨μ_i, e_w⟩)
        Returns log β_{i,w} unnormalized, shape: [n_topics, V]
        """
        device = mu.device
        dtype = mu.dtype
        kappa = kappa.to(device=device, dtype=dtype)
        vec = vec.to(device=device, dtype=dtype)

        mu = torch.nn.functional.normalize(mu, dim=-1)
        vec = torch.nn.functional.normalize(vec, dim=-1)

        dot = torch.mm(mu, vec.t())  # [n_topics, V]
        log_unnorm = kappa.unsqueeze(1) * dot  # [n_topics, V]

        d = mu.size(-1)
        nu = d / 2.0 - 1.0  # e.g., 149.0 for d=300

        # Use asymptotic expansion for large ν or moderate κ:
        # log I_ν(κ) ≈ κ + (ν - 0.5) * log(κ / (ν + sqrt(ν² + κ²))) - sqrt(ν² + κ²) + 0.5*log(2π)
        # Also used in Hyperspherical VAE (Davidson et al. 2018)

        # Compute sqrt(ν² + κ²) safely
        nu_t = torch.full_like(kappa, fill_value=nu)  # [n_topics]
        sqrt_term = torch.sqrt(nu_t ** 2 + kappa ** 2)  # [n_topics]

        # Asymptotic approximation of log I_ν(κ)
        # log I_ν(κ) ≈ κ - sqrt(ν² + κ²) + (ν - 0.5) * log( κ / (ν + sqrt(ν² + κ²)) ) + 0.5*log(2π)
        log_I_approx = (
                kappa
                - sqrt_term
                + (nu_t - 0.5) * torch.log(kappa / (nu_t + sqrt_term).clamp(min=1e-12))
                + 0.5 * math.log(2 * math.pi)
        )

        # Optional: for very small κ (e.g., < 1e-3), use series expansion:
        # I_ν(κ) ≈ (κ/2)^ν / Γ(ν+1)  →  log I_ν(κ) ≈ ν*log(κ/2) - lgamma(ν+1)
        small_kappa = kappa < 1e-3
        if small_kappa.any():
            log_gamma = torch.lgamma(torch.tensor([nu + 1.0], device=device, dtype=dtype))
            log_I_series = nu * torch.log(kappa[small_kappa].clamp(min=1e-12) / 2.0) - log_gamma
            log_I_approx[small_kappa] = log_I_series.squeeze()

        # Now use log_I_approx as log I_ν(κ)
        log_I_nu_kappa = log_I_approx  # [n_topics]

        # Compute log normalization constant
        log_C = (nu) * torch.log(kappa.clamp(min=1e-12)) \
                - (d / 2.0) * math.log(2 * math.pi) \
                - log_I_nu_kappa  # [n_topics]

        logp = log_unnorm - log_C.unsqueeze(1)  # [n_topics, V]
        return logp