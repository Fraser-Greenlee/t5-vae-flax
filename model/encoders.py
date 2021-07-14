import logging
import jax.numpy as jnp
import flax.linen as nn

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    '''
        Converts N hidden tokens into N seperate latent codes.
    '''
    latent_token_size: int
    n_latent_tokens: int

    @nn.compact
    def __call__(self, encoding):
        latent_tokens = nn.Dense(self.latent_token_size)(encoding)
        raw_latent_code = latent_tokens[:, : self.n_latent_tokens, :]
        # TODO does this just apply tanh to each latent token? Or across the whole batch
        latent_code = jnp.tanh(raw_latent_code)
        return latent_code  # (batch, latent_tokens_per_sequence, latent_token_dim)


VAE_ENCODER_MODELS = {
    '': Encoder,
}
