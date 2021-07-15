import logging
import flax.linen as nn

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    '''
        Converts latent code -> transformer encoding.
    '''
    dim_model: int
    n_latent_tokens: int

    @nn.compact
    def __call__(self, latent_code):  # (batch, latent_tokens_per_sequence, latent_token_dim)
        raw_latent_tokens = nn.Dense(self.dim_model)(latent_code)
        latent_tokens = nn.LayerNorm()(raw_latent_tokens)
        return latent_tokens  # (batch, latent_tokens_per_sequence, dim_model)


VAE_DECODER_MODELS = {
    '': Decoder,
}
