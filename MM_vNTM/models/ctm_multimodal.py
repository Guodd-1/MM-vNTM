"Based on: https://github.com/gonzalezf/multimodal_neural_topic_modeling/"
import datetime
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import torch
from scipy.special import softmax
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from ..utils.early_stopping.early_stopping import EarlyStopping
# decoder network
from ..networks.decoding_network_multimodal import DecoderNetworkMultimodal
from ..networks.decoding_network_multimodal import DecoderNetworkMultimodal_ETM
from ..networks.decoding_network_multimodal import DecoderNetworkMultimodal_vMFMix
import torch.nn as nn

class CTM_Multimodal:
    """Class to train the contextualized topic model. This is the more general class that we are keeping to
    avoid braking code, users should use the two subclasses ZeroShotTM and CombinedTm to do topic modeling.

    :param bow_size: int, dimension of input
    :param contextual_size: int, dimension of input that comes from BERT embeddings
    :param inference_type: string, you can choose between the contextual model and the combined model
    :param n_components: int, number of topic components, (default 10)
    :param model_type: string, 'prodLDA' or 'LDA' (default 'prodLDA')
    :param hidden_sizes: tuple, length = n_layers, (default (100, 100))
    :param activation: string, 'softplus', 'relu', (default 'softplus')
    :param dropout: float, dropout to use (default 0.2)
    :param learn_priors: bool, make priors a learnable parameter (default True)
    :param batch_size: int, size of batch to use for training (default 64)
    :param lr: float, learning rate to use for training (default 2e-3)
    :param momentum: float, momentum to use for training (default 0.99)
    :param solver: string, optimizer 'adam' or 'sgd' (default 'adam')
    :param num_epochs: int, number of epochs to train for, (default 100)
    :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
    :param num_data_loader_workers: int, number of data loader workers (default cpu_count). set it to 0 if you are using Windows
    :param label_size: int, number of total labels (default: 0)
    :param loss_weights: dict, it contains the name of the weight parameter (key) and the weight (value) for each loss.
    It supports only the weight parameter beta for now. If None, then the weights are set to 1 (default: None).

    """

    def __init__(self, bow_size, contextual_size, inference_type="combined", n_components=10, model_type='prodLDA',
                 decoder_type='original',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2, learn_priors=True, batch_size=64,
                 lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, reduce_on_plateau=False,
                 num_data_loader_workers=mp.cpu_count(), label_size=0, loss_weights=None, img_enc_dim=0, kappa1=10,
                 kappa2=10, temp=10, embedding_matrix=None):

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print('Initializing CTM  Multimodal class')
        if self.__class__.__name__ == "CTM":
            raise Exception("You cannot call this class. Use ZeroShotTM or CombinedTM")

        assert isinstance(bow_size, int) and bow_size > 0, \
            "input_size must by type int > 0."
        assert isinstance(n_components, int) and bow_size > 0, \
            "n_components must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'], \
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(learn_priors, bool), "learn_priors must be boolean."
        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(momentum, float) and 0 < momentum <= 1, \
            "momentum must be 0 < float <= 1."
        assert solver in ['adam', 'sgd'], "solver must be 'adam' or 'sgd'."
        assert isinstance(reduce_on_plateau, bool), \
            "reduce_on_plateau must be type bool."
        assert isinstance(num_data_loader_workers, int) and num_data_loader_workers >= 0, \
            "num_data_loader_workers must by type int >= 0. set 0 if you are using windows"

        self.bow_size = bow_size
        self.n_components = n_components
        self.model_type = model_type
        self.decoder_type = decoder_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.contextual_size = contextual_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.training_doc_topic_distributions = None
        self.latent_z = None
        self.img_enc_dim = img_enc_dim
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.temp = temp
        self.embedding_matrix = embedding_matrix

        if loss_weights:
            self.weights = loss_weights
        else:
            # self.weights = {"KL": 1, 'EL':50} # We will use 1 as the CTM paper # beta and KL is the same, for the CTM and M3L paper respectively. # self.weights = {"beta": 1}
            self.weights = {"KL": 1, "BoW_loss": 1, 'IMG_loss': 1, 'TXT_loss': 1}

        print('Current loss weights: ', self.weights)

        if self.decoder_type == 'original':
            self.model = DecoderNetworkMultimodal(
                bow_size, self.contextual_size, inference_type, n_components, model_type, hidden_sizes, activation,
                dropout, learn_priors, label_size=label_size, img_enc_dim=img_enc_dim,
                kappa1=self.kappa1, kappa2=self.kappa2, temp=self.temp, embedding_matrix=self.embedding_matrix)
        elif self.decoder_type == 'ETM':
            self.model = DecoderNetworkMultimodal_ETM(
                bow_size, self.contextual_size, inference_type, n_components, model_type, hidden_sizes, activation,
                dropout, learn_priors, label_size=label_size, img_enc_dim=img_enc_dim,
                kappa1=self.kappa1, kappa2=self.kappa2, temp=self.temp, embedding_matrix=self.embedding_matrix)
        elif self.decoder_type == 'vMFMix':
            self.model = DecoderNetworkMultimodal_vMFMix(
                bow_size, self.contextual_size, inference_type, n_components, model_type, hidden_sizes, activation,
                dropout, learn_priors, label_size=label_size, img_enc_dim=img_enc_dim,
                kappa1=self.kappa1, kappa2=self.kappa2, temp=self.temp, embedding_matrix=self.embedding_matrix)

        self.early_stopping = None

        # initialize optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)

        # initialize learning rate scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_loss_train = float('inf')

        # training attributes
        self.model_dir = None
        self.nn_epoch = None

        # validation attributes
        self.validation_data = None

        # learned topics
        self.best_components = None
        self.best_components_img = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        self.model = self.model.to(self.device)

    def _rl_loss(self, true_word_dists, pred_word_dists):
        # Reconstruction term (BoW)

        RL = -torch.sum(true_word_dists * torch.log(pred_word_dists + 1e-10), dim=1)
        return RL

    def _rl_loss_img_cosine(self, true_word_dists, pred_word_dists):
        target = torch.full((self.batch_size,), 1).cuda()
        loss_img = nn.CosineEmbeddingLoss(reduction='none')
        output = loss_img(true_word_dists, pred_word_dists, target)
        RL = output.reshape(output.shape[0], 1)

        return RL

    def _loss(self, inputs, word_dists, kl_div,
              input_img_features=None, predicted_img_features=None,
              X_textual_embeddings=None, predicted_textual_features=None,
              img_reconstruction=True, txt_reconstruction=True):

        # KL term
        KL = kl_div  # ← kl_div is already a [N,] tensor

        # Reconstruction term (BoW)
        RL = self._rl_loss(inputs, word_dists)

        # image reconstruction term
        if img_reconstruction:
            RL_img = self._rl_loss_img_cosine(input_img_features, predicted_img_features)
            RL_img = RL_img.reshape(RL_img.shape[0])  # reshape to 1D
        else:
            RL_img = None

        # text reconstruction term
        if txt_reconstruction:
            RL_text = self._rl_loss_img_cosine(X_textual_embeddings, predicted_textual_features)
            RL_text = RL_text.reshape(RL_text.shape[0])  # reshape to 1D
        else:
            RL_text = None

        return KL, RL, RL_img, RL_text

    def fit(self, train_dataset, validation_dataset=None, save_dir=None, verbose=False, patience=5, delta=0,
            n_samples=20, multimodal=True, tensorboard=False, txt_reconstruction=True, img_reconstruction=True):

        # Save model dir and token mapping (idx → token)
        self.model_dir = save_dir
        self.idx2token = train_dataset.idx2token
        train_data = train_dataset
        self.validation_data = validation_dataset

        # Initialize early stopping if validation set is provided
        if self.validation_data is not None:
            self.early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=save_dir, delta=delta)

        # Create training data loader
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_data_loader_workers,
            drop_last=True  # drop incomplete last batch
        )

        # Initialize training metrics (updated later in _train_epoch; kept for completeness)
        train_loss = 0
        samples_processed = 0
        train_loss_rl_bow = 0
        train_loss_rl_img = 0

        # Initialize progress bar
        pbar = tqdm(self.num_epochs, position=0, leave=True)

        # Initialize TensorBoard writer if enabled
        if tensorboard:
            writer = SummaryWriter()

        # ============ Main training loop ============
        for epoch in range(self.num_epochs):
            # print("-"*10, "Epoch", epoch+1, "-"*10)
            self.nn_epoch = epoch  # record current epoch (used for model saving)

            # Train current epoch
            s = datetime.datetime.now()
            # Call _train_epoch, returns processed sample count and average losses
            sp, train_loss, train_loss_rl_bow, train_loss_rl_img, train_loss_kl, train_loss_rl_text = self._train_epoch(
                train_loader,
                multimodal=multimodal,
                txt_reconstruction=txt_reconstruction,
                img_reconstruction=img_reconstruction,
                epoch=epoch  # for dynamic weight adjustment
            )
            samples_processed += sp
            e = datetime.datetime.now()
            pbar.update(1)

            # ============ Validation stage (optional) ============
            if self.validation_data is not None:
                # Create validation loader
                validation_loader = DataLoader(
                    self.validation_data,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_data_loader_workers,
                    drop_last=True
                )

                # Run validation (note: _validation does NOT support multimodal reconstruction, only BoW + KL)
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validation(validation_loader)
                e = datetime.datetime.now()

                # Print validation results
                if verbose:
                    print("Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                        epoch + 1, self.num_epochs, val_samples_processed,
                        len(self.validation_data) * self.num_epochs, val_loss, e - s))

                # Update progress bar description
                pbar.set_description(
                    "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tValid Loss: {}\tTime: {}".format(
                        epoch + 1, self.num_epochs, samples_processed,
                        len(train_data) * self.num_epochs, train_loss, val_loss, e - s))

                # Check early stopping condition
                self.early_stopping(val_loss, self)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

            else:
                # If no validation set, save current parameters as best (actually last epoch)
                self.best_components = self.model.beta  # topic-word matrix
                self.best_components_img = self.model.beta_img  # topic-image matrix
                self.best_components_text = self.model.beta_text_features  # topic-text matrix

                if save_dir is not None:
                    self.save(save_dir)  # save model (experimental)

            # Update progress bar: display losses
            pbar.set_description(
                "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\t"
                "Train Loss: {:.4f}\tTime: {}\t"
                "Train RL BOW Loss: {:.4f}\t"
                "Train RL IMG Loss: {:.4f}\t"
                "Train KL Loss: {:.8f}\t"
                "Train RL text: {:.4f}\t".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(train_data) * self.num_epochs,
                    train_loss, e - s,
                    train_loss_rl_bow, train_loss_rl_img,
                    train_loss_kl, train_loss_rl_text
                )
            )

            # Log to TensorBoard
            if tensorboard:
                writer.add_scalar('Loss/total', train_loss, epoch)
                writer.add_scalar('Loss/rl_bow', train_loss_rl_bow, epoch)
                writer.add_scalar('Loss/rl_img', train_loss_rl_img, epoch)
                writer.add_scalar('Loss/rl_text', train_loss_rl_text, epoch)
                writer.add_scalar('Loss/kl_divergence', train_loss_kl, epoch)

        # Close progress bar
        pbar.close()

        # ============ Post-training: infer document-topic distributions ============
        # Perform multiple sampling (n_samples times), average θ as final result
        self.latent_z, self.training_doc_topic_distributions = self.get_doc_topic_distribution(train_dataset, n_samples)

    def _train_epoch(self, loader, multimodal=True, txt_reconstruction=False, img_reconstruction=True, epoch=None):
        """Train for one epoch.

        Args:
            loader: PyTorch DataLoader providing training batches.
            multimodal: whether to enable multimodal (image + text), default True.
            txt_reconstruction: whether to include text embedding reconstruction loss, default False.
            img_reconstruction: whether to include image embedding reconstruction loss, default True.
            epoch: current epoch number (used for dynamic loss weight adjustment, effective only in epoch=0).
        """
        self.model.train()  # set model to training mode (enable dropout, etc.)

        # Initialize accumulators
        samples_processed = 0  # number of samples processed
        train_loss = 0  # total loss
        train_loss_rl_bow = 0  # BoW reconstruction loss
        train_loss_rl_img = 0  # image reconstruction loss
        train_loss_rl_txt = 0  # text reconstruction loss
        train_loss_kl = 0  # KL divergence loss

        # Store dynamic weights for epoch 0 (to align magnitude with BoW loss)
        list_txt_weight_epoch_0 = []
        list_img_weight_epoch_0 = []

        # Iterate over each batch
        for batch_samples in loader:
            # Extract BoW features: shape [batch_size, vocab_size]
            X_bow = batch_samples['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)

            # Extract contextual text embeddings (e.g., SBERT/CLIP): [batch_size, D_text]
            X_contextual = batch_samples['X_contextual']

            # If multimodal is enabled, load image embeddings and indices
            if multimodal:
                X_image_embeddings = batch_samples['X_image_embeddings']  # [batch_size, D_img]
                X_image_embeddings_index = batch_samples['X_image_embeddings_index']  # list of image IDs

            # Load labels (if available)
            if "labels" in batch_samples.keys():
                labels = batch_samples["labels"]
                labels = labels.reshape(labels.shape[0], -1)
                labels = labels.to(self.device)
            else:
                labels = None

            # Move data to GPU (if available)
            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()
                if multimodal:
                    X_image_embeddings = X_image_embeddings.cuda()
                    # X_image_embeddings_index is list of strings — cannot call .cuda()

            # Zero gradients
            self.model.zero_grad()

            # ========== Forward pass ==========
            # Call DecoderNetworkMultimodal.forward()
            # Note: here txt_reconstruction=True, img_reconstruction=True are hardcoded,
            # but actual loss inclusion is controlled by parameters (no harm, just redundant)
            kl_div, word_dists, estimated_labels, img_feature_dists, predicted_textual_features = self.model(
                x=X_bow,
                x_bert=X_contextual,
                labels=labels,
                X_image_embeddings=X_image_embeddings,
                txt_reconstruction=True,
                img_reconstruction=True)

            # ========== Compute losses ==========
            kl_loss, rl_loss, rl_img_loss, rl_text_loss = self._loss(
                inputs=X_bow,
                word_dists=word_dists,
                kl_div=kl_div,
                input_img_features=X_image_embeddings,
                predicted_img_features=img_feature_dists,
                X_textual_embeddings=X_contextual,
                predicted_textual_features=predicted_textual_features,
                img_reconstruction=img_reconstruction,  # control image loss
                txt_reconstruction=txt_reconstruction  # control text loss
            )

            # ========== Build total loss ==========
            # Base loss: BoW reconstruction + KL divergence
            loss = self.weights["BoW_loss"] * rl_loss + self.weights["KL"] * kl_loss

            # If image reconstruction is enabled
            if img_reconstruction:
                # Dynamically adjust image loss weight in epoch 0 (align with BoW scale)
                if epoch == 0 and self.weights["IMG_loss"] != -1:
                    # Compute scaling factor: BoW total loss / image total loss
                    self.final_weight_img_loss = self.weights["IMG_loss"] * (
                            self.weights["BoW_loss"] * rl_loss.sum().item() / rl_img_loss.sum().item()
                    )
                    list_img_weight_epoch_0.append(self.final_weight_img_loss)
                # If user sets weight = -1, fix weight to 1 (skip dynamic scaling)
                if self.weights["IMG_loss"] == -1:
                    self.final_weight_img_loss = 1

                # Add weighted image loss to total loss
                loss += self.final_weight_img_loss * rl_img_loss

            # If text reconstruction is enabled
            if txt_reconstruction:
                # train_loss_rl_txt += self.weights["TXT_loss"] * rl_text_loss.sum().item()  # original
                if epoch == 0 and self.weights["TXT_loss"] != -1:
                    self.final_weight_txt_loss = self.weights["TXT_loss"] * (
                            self.weights["BoW_loss"] * rl_loss.sum().item() / rl_text_loss.sum().item())
                    list_txt_weight_epoch_0.append(self.final_weight_txt_loss)
                # else:
                # final_weight_txt_loss is the mean of weights computed in epoch 0 (stored in list_txt_weight_epoch_0)

                # If user specifies -1, fix weight to 1
                if self.weights["TXT_loss"] == -1:
                    self.final_weight_txt_loss = 1

                # loss += self.weights["TXT_loss"] * rl_text_loss  # old
                loss += self.final_weight_txt_loss * rl_text_loss  # new

            # ========== Backward pass and optimization ==========
            loss = loss.sum()  # sum over batch to get scalar
            loss.backward()
            self.optimizer.step()

            # ========== Accumulate training metrics ==========
            batch_size = X_bow.size()[0]
            samples_processed += batch_size
            train_loss += loss.item()
            train_loss_rl_bow += self.weights["BoW_loss"] * rl_loss.sum().item()
            if img_reconstruction:
                train_loss_rl_img += self.final_weight_img_loss * rl_img_loss.sum().item()

            if txt_reconstruction:
                train_loss_rl_txt += self.final_weight_txt_loss * rl_text_loss.sum().item()

            train_loss_kl += self.weights["KL"] * kl_loss.sum().item()

        # ========== Determine final dynamic weights at epoch 0 ==========
        if epoch == 0 and self.weights["TXT_loss"] != -1:
            self.final_weight_txt_loss = np.mean(list_txt_weight_epoch_0)
            print('MEAN New weight for txt loss', self.final_weight_txt_loss)

        if epoch == 0 and self.weights["IMG_loss"] != -1:
            self.final_weight_img_loss = np.mean(list_img_weight_epoch_0)
            print('MEAN New weight for img loss', self.final_weight_img_loss)

        # ========== Compute average per-sample losses ==========
        train_loss /= samples_processed
        train_loss_rl_bow /= samples_processed
        train_loss_rl_img /= samples_processed
        train_loss_kl /= samples_processed
        train_loss_rl_txt /= samples_processed

        # Return: processed samples, average losses
        return samples_processed, train_loss, train_loss_rl_bow, train_loss_rl_img, train_loss_kl, train_loss_rl_txt

    def _validation(self, loader):
        """Validate for one epoch.

        This method evaluates model performance on the validation set.
        **Only BoW reconstruction + KL divergence are computed**,
        **image/text embedding reconstruction losses are NOT included** (even if enabled during training).
        """
        self.model.eval()  # set model to evaluation mode (disable dropout, etc.)

        val_loss = 0  # accumulated validation loss
        samples_processed = 0  # number of processed samples

        # Iterate over validation batches
        for batch_samples in loader:
            # Extract BoW features [batch_size, vocab_size]
            X_bow = batch_samples['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)

            # Extract contextual text embeddings [batch_size, D_text]
            X_contextual = batch_samples['X_contextual']

            # Handle labels (if present)
            if "labels" in batch_samples.keys():
                labels = batch_samples["labels"]
                labels = labels.to(self.device)
                labels = labels.reshape(labels.shape[0], -1)
            else:
                labels = None

            # Move data to GPU (if enabled)
            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # ========== Forward pass (no gradient) ==========
            # Note: image embeddings NOT passed; img/txt reconstruction NOT enabled,
            # so model.forward() returns only (kl_div, word_dists, estimated_labels)
            self.model.zero_grad()  # safe to call in eval mode
            kl_div, word_dists, estimated_labels = self.model(X_bow, X_contextual, labels)
            # img_feature_dists and predicted_textual_features are None

            # ========== Compute loss ==========
            # Call _loss with only BoW-related args → img/txt reconstruction defaults to False
            kl_loss, rl_loss = self._loss(
                inputs=X_bow,
                word_dists=word_dists,
                kl_div=kl_div
            )

            loss = self.weights["KL"] * kl_loss + self.weights["BoW_loss"] * rl_loss
            loss = loss.sum()  # sum over batch

            # If labels exist, add classification loss
            if labels is not None:
                target_labels = torch.argmax(labels, 1)  # assuming one-hot
                label_loss = torch.nn.CrossEntropyLoss()(estimated_labels, target_labels)
                loss += label_loss

            # ========== Accumulate metrics ==========
            batch_size = X_bow.size(0)
            samples_processed += batch_size
            val_loss += loss.item()

        # Compute average per-sample validation loss
        val_loss /= samples_processed

        return samples_processed, val_loss

    def get_thetas(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics.
        Includes multiple sampling to reduce variation via the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of samples to collect to estimate the final distribution (more is better).
        """
        return self.get_doc_topic_distribution(dataset, n_samples=n_samples)

    def get_doc_topic_distribution(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics.
        Includes multiple sampling to reduce variation via the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of samples to collect to estimate the final distribution (more is better).
        """
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)
        pbar = tqdm(n_samples, position=0, leave=True)
        final_z = []
        final_thetas = []
        for sample_index in range(n_samples):
            with torch.no_grad():
                collect_z = []
                collect_theta = []
                for batch_samples in loader:
                    # batch_size x vocab_size
                    if "X_bow" in batch_samples.keys():
                        X_bow = batch_samples['X_bow']
                        X_bow = X_bow.reshape(X_bow.shape[0], -1)
                    else:
                        # Here we create a zero vector of vocabulary size
                        X_bow = torch.zeros(batch_samples['X_image_embeddings'].shape[0], self.vocab_size)
                    X_bow = X_bow.to(self.device)
                    if "X_contextual" in batch_samples.keys():
                        X_contextual = batch_samples['X_contextual']
                    else:
                        # Assumption: model trained in multimodal mode,
                        # inference may be on text-only, image-only, or multimodal.
                        # Create zero vector of contextual size
                        X_contextual = torch.zeros(batch_samples['X_image_embeddings'].shape[0], self.contextual_size)
                    X_contextual = X_contextual.to(self.device)
                    if "X_image_embeddings" in batch_samples.keys():
                        X_image_embeddings = batch_samples['X_image_embeddings']
                    else:
                        # Create zero vector of image embedding size (important for inference)
                        X_image_embeddings = torch.zeros(batch_samples['X_contextual'].shape[0], self.img_enc_dim)
                    X_image_embeddings = X_image_embeddings.to(self.device)

                    if "X_image_embeddings_index" in batch_samples.keys():
                        X_image_embeddings_index = batch_samples['X_image_embeddings_index']
                        # X_image_embeddings_index is list of strings — cannot call .cuda()

                    # batch_samples.keys() → dict_keys(['X_bow', 'X_contextual', 'X_image_embeddings', 'X_image_embeddings_index'])

                    if "labels" in batch_samples.keys():
                        labels = batch_samples["labels"]
                        labels = labels.to(self.device)
                        labels = labels.reshape(labels.shape[0], -1)
                    else:
                        labels = None

                    # forward pass
                    self.model.zero_grad()
                    z1, theta1 = self.model.get_theta(X_bow, X_contextual, labels, X_image_embeddings)
                    z1 = z1.cpu().numpy().tolist()
                    theta1 = theta1.cpu().numpy().tolist()

                    collect_z.extend(z1)
                    collect_theta.extend(theta1)

                pbar.update(1)
                pbar.set_description("Sampling: [{}/{}]".format(sample_index + 1, n_samples))

                final_z.append(np.array(collect_z))
                final_thetas.append(np.array(collect_theta))
        pbar.close()
        return np.sum(final_z, axis=0) / n_samples, np.sum(final_thetas, axis=0) / n_samples

    def get_most_likely_topic(self, doc_topic_distribution):
        """ Get the most likely topic for each document

        :param doc_topic_distribution: ndarray representing the topic distribution of each document
        """
        return np.argmax(doc_topic_distribution, axis=0)

    def get_topics(self, k=10):  ## original CTM
        """
        Retrieve topic words.

        :param k: int, number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        component_dists = self.best_components
        topics = defaultdict(list)
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics[i] = component_words
        return topics

    # Get closest image embeddings based on similarity to beta_img
    def get_topics_image_embeddings_beta_img(self, image_embeddings, image_embeddings_index, k=10):
        '''
        topics = defaultdict(list)
        for current_topic_id in range(self.n_components):
            top_vectors = self.get_list_of_closest_image_embeddings_to_beta_img(current_topic_id, image_embeddings, image_embeddings_index, k)
            topics[current_topic_id] = top_vectors
        return topics
        '''

        topics = []
        for current_topic_id in range(self.n_components):
            top_vectors = self.get_list_of_closest_image_embeddings_to_beta_img(current_topic_id, image_embeddings,
                                                                                image_embeddings_index, k)
            topics.append(top_vectors)
        return topics

    # Get top-k image embeddings per topic
    def get_list_of_closest_image_embeddings_to_beta_img(self, topic_id, image_embeddings, image_embeddings_index,
                                                         k=10):  #
        """
        Return the images most similar to the learned topic_image_embedding feature.
        :param k: int, number of image features to return per topic, default 10.
        """

        # representative vector for this topic
        representative_img_feature_vector = self.best_components_img[topic_id]

        # Cosine similarity may return values >1 due to numerical precision (see PyTorch issue #78064).
        # Ref: https://github.com/pytorch/pytorch/issues/78064
        # Ref: CLIP Colab also normalizes vectors before cosine similarity.
        # To avoid issues, we use torch.nn.functional.cosine_similarity (handles normalization internally).

        # New solution — safe
        query_similarity = torch.nn.functional.cosine_similarity(representative_img_feature_vector.unsqueeze(0),
                                                                 image_embeddings, dim=1)

        # old solution — unsafe (values can exceed 1 due to lack of normalization)
        # query_similarity = representative_img_feature_vector @ image_embeddings.T

        query_similarity = query_similarity.cpu().detach().numpy()

        list_closest_image_embeddings = []
        for i in range(1, k + 1):
            image_index = np.argsort(query_similarity, axis=0)[-i]
            current_image_embedding = image_embeddings[image_index]
            list_closest_image_embeddings.append(current_image_embedding)

        return list_closest_image_embeddings

    def get_topic_lists(self, k=10):
        """
        Retrieve lists of topic words.
        :param k: (int) number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."

        component_dists = self.best_components
        topics = []
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics.append(component_words)
        return topics

    def get_topic_word_matrix(self):
        """
        Return the topic-word matrix (dimensions: number of topics x vocab size).
        If model_type is LDA, the matrix is normalized; otherwise unnormalized.
        """
        return self.model.topic_word_matrix.cpu().detach().numpy()

    def get_topic_word_distribution(self):
        """
        Return the topic-word distribution (dimensions: number of topics x vocab size).
        """
        mat = self.get_topic_word_matrix()
        return softmax(mat, axis=1)

    def get_word_distribution_by_topic_id(self, topic):
        """
        Return the word probability distribution of a topic sorted by probability.

        :param topic: id of the topic (int)

        :returns list of tuples (word, probability) sorted descending by probability
        """
        if topic >= self.n_components:
            raise Exception('Topic id must be lower than the number of topics')
        else:
            wd = self.get_topic_word_distribution()
            t = [(word, wd[topic][idx]) for idx, word in self.idx2token.items()]
            t = sorted(t, key=lambda x: -x[1])
        return t

    def get_predicted_topics(self, dataset, n_samples):
        """
        Return a list of predicted topic IDs for each document (length = number of documents).

        :param dataset: CTMDataset for inference
        :param n_samples: number of theta samples
        :return: predicted topics (list of int)
        """
        predicted_topics = []
        thetas = self.get_doc_topic_distribution(dataset, n_samples)

        for idd in range(len(dataset)):
            predicted_topic = np.argmax(thetas[idd] / np.sum(thetas[idd]))
            predicted_topics.append(predicted_topic)
        return predicted_topics

    # Identify top image embeddings based on training_doc_topic_distributions (inference on training set)
    def get_top_image_embeddings(self, current_topic_id, training_doc_topic_distributions, image_embeddings, top_n=10):
        # Get top documents with highest topic probability
        current_list_indices = (-training_doc_topic_distributions).argsort(axis=0).T[current_topic_id]
        return torch.Tensor(image_embeddings[current_list_indices[:top_n]])

    # Get list of top image embeddings for all topics
    def get_list_top_image_embeddings(self, training_doc_topic_distributions, image_embeddings, top_n=10):
        list_top_image_embeddings = []
        for current_topic_id in range(self.n_components):
            list_top_image_embeddings.append(
                self.get_top_image_embeddings(current_topic_id, training_doc_topic_distributions, image_embeddings,
                                              top_n))
        return list_top_image_embeddings


class ZeroShotTM(CTM_Multimodal):
    def __init__(self, **kwargs):
        inference_type = "zeroshot"
        super().__init__(**kwargs, inference_type=inference_type)


class CombinedTM(CTM_Multimodal):
    def __init__(self, **kwargs):
        inference_type = "combined"
        super().__init__(**kwargs, inference_type=inference_type)