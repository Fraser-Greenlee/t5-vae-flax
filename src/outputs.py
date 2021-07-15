from typing import Optional, Tuple

import flax
import jaxlib.xla_extension as jax_xla

from transformers.file_utils import ModelOutput


@flax.struct.dataclass
class TransformerVaeOutput(ModelOutput):
    """
    Base class for a Transformer-VAE's outputs.

    Args:
        latent_codes (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_latent_tokens, latent_token_size)`):
            Latent codes representing encoded sequences.
        remade_encoder_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_tokens, model_dim)`):
            Reconstructed encoder hidden states representing sequences.

    (std Seq2Seq) Args:
        logits (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`tuple(tuple(jax_xla.DeviceArray))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(jax_xla.DeviceArray)` of length :obj:`config.n_layers`, with each tuple having 2
            tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
            tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        last_hidden_state (:obj:`tuple(jax_xla.DeviceArray)`:
            Last model hidden state.
        decoder_hidden_states (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for the output of the embeddings + one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(jax_xla.DeviceArray)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jax_xla.DeviceArray` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    logits: jax_xla.DeviceArray = None
    latent_codes: jax_xla.DeviceArray = None
    remade_encoder_hidden_state: jax_xla.DeviceArray = None
    # seq2seq
    past_key_values: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None
    decoder_hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    decoder_attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    cross_attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    last_hidden_state: Optional[jax_xla.DeviceArray] = None
    encoder_last_hidden_state: Optional[jax_xla.DeviceArray] = None
    encoder_hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    encoder_attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
