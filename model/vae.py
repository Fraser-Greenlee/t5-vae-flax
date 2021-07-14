import jax.numpy as jnp
import flax.linen as nn

from model.encoders import VAE_ENCODER_MODELS
from model.decoders import VAE_DECODER_MODELS
from model.config import T5VaeConfig


class VAE(nn.Module):
    # see https://github.com/google/flax#what-does-flax-look-like
    """
        An MMD-VAE used with encoder-decoder models.
        Encodes all token encodings into a single latent & spits them back out.
    """
    config: T5VaeConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.encoder = VAE_ENCODER_MODELS[self.config.vae_encoder_model](self.config.latent_token_size, self.config.n_latent_tokens)
        self.decoder = VAE_DECODER_MODELS[self.config.vae_decoder_model](self.config.t5.d_model,  self.config.n_latent_tokens)

    def __call__(self, encoding=None, latent_codes=None):
        latent_codes = self.encode(encoding)
        return self.decode(latent_codes), latent_codes

    def encode(self, encoding):
        return self.encoder(encoding)

    def decode(self, latent):
        return self.decoder(latent)
