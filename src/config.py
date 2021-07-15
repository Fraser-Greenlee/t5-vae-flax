import copy
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig, T5Config

from t5_vae_flax.src.encoders import VAE_ENCODER_MODELS
from t5_vae_flax.src.decoders import VAE_DECODER_MODELS
from t5_vae_flax.src.utils import assertEqual, assertIn

logger = logging.get_logger(__name__)


class T5VaeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of :class:`FlaxT5VAE`.
    It is used to instantiate a T5-VAE model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the T5 `t5-vae-base architecture.

    To be able to use `transformer.trainer.Trainer` we need some specific training logic & config in the model.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        n_latent_tokens (:obj:`int`, `optional`, defaults to 6):
            Number of latent tokens (must be less than seq length).
        latent_token_size (:obj:`int`, `optional`, defaults to 32):
            Number of dimensions to use for each latent token.
        t5_name (:obj:`str`, `optional`, defaults to t5-base):
            Name of the Transformer model to use as a decoder.
        block_size (:obj:`int`, `optional`, defaults to 60):
            NOTE: Every input sequence must be padded to be equal to this length.
    """
    model_type = "transformer_vae"
    is_composition = True

    def __init__(
        self,
        t5_model_name_or_path=None,
        n_latent_tokens=6,  # set to -1 for full sequence
        latent_token_size=32,
        vae_encoder_model='',
        vae_decoder_model='',
        block_size=60,
        decoder_start_token_id=0,
        cache_dir=None,
        tie_word_embeddings=True,
        # T5 config
        t5=dict(),
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        gradient_checkpointing=False,
        # end
        **kwargs,
    ):
        assertIn(vae_encoder_model, VAE_ENCODER_MODELS.keys(), "Unexpected VAE encoder.")
        assertIn(vae_decoder_model, VAE_DECODER_MODELS.keys(), "Unexpected VAE decoder.")

        super().__init__(**kwargs)

        self.set_seq_size = block_size

        # VAE
        self.vae_encoder_model = vae_encoder_model
        self.vae_decoder_model = vae_decoder_model

        self.latent_token_size = latent_token_size
        assert(n_latent_tokens <= self.set_seq_size, 'Cannot use more latent tokens than input tokens.')
        self.n_latent_tokens = n_latent_tokens
        self.use_cache = use_cache

        # T5
        if t5_model_name_or_path:
            self.t5 = AutoConfig.from_pretrained(t5_model_name_or_path, cache_dir=cache_dir)
            assertEqual(self.t5.model_type, "t5", "Need t5 model type for transformer_decoder.")
            self.t5.decoder_start_token_id = decoder_start_token_id
        elif t5:
            # use for loading a config
            self.t5 = T5Config(**t5)
        else:
            self.t5 = T5Config(
                vocab_size=vocab_size,
                d_model=d_model,
                d_kv=d_kv,
                d_ff=d_ff,
                num_layers=num_layers,
                num_decoder_layers=num_decoder_layers,
                num_heads=num_heads,
                relative_attention_num_buckets=relative_attention_num_buckets,
                dropout_rate=dropout_rate,
                layer_norm_epsilon=layer_norm_epsilon,
                initializer_factor=initializer_factor,
                feed_forward_proj=feed_forward_proj,
                is_encoder_decoder=is_encoder_decoder,
                use_cache=use_cache,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                gradient_checkpointing=gradient_checkpointing,
                **kwargs
            )

        if self.t5.d_model < self.latent_token_size:
            raise Exception('Using larger latent token dimension then T5 hidden dimension.')

        # Add t5 config options
        self.tie_word_embeddings = tie_word_embeddings
        self.t5.tie_word_embeddings = self.tie_word_embeddings
        self.t5.use_cache = self.use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = self.t5.decoder_start_token_id

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        output['t5'] = self.t5.to_dict()
        return output
