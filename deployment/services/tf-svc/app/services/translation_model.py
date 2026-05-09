# Pylance / Pyright suppressions for known-false-positive Keras stubs.
# Pylance's bundled TensorFlow stubs declare `Layer.call(self, inputs)`
# and `Layer.__call__(self, inputs)` - tight signatures that real
# Keras layers routinely override (Keras's __call__ dispatches via
# *args, **kwargs). Same story for Model.call(inputs, training, mask).
# Subclassing those classes with the standard call(self, query, key,
# value, mask, training) pattern triggers reportIncompatibleMethodOverride
# and reportCallIssue. Mypy strict (the CLI gate) does not flag these
# because it treats the imports as Any; the runtime behavior is correct
# (verified end-to-end by load_weights succeeding with zero key
# mismatches). Suppressing only these specific rules keeps every other
# real type check active.
# pyright: reportIncompatibleMethodOverride=false, reportCallIssue=false

"""
Transformer architecture for the EN-ES translation endpoint.

WHAT THIS FILE IS:
    JUST the model class + its building-block layers + two mask
    helpers. No I/O, no MLflow, no tokenization, no decoding loop.
    The trained weights live in the registry as
    `translation_transformer.weights.h5`; this class is the
    architecture spec those weights deserialize INTO.

    Loading flow:
        1. Build the architecture          (this file's classes)
        2. Run a dummy forward pass        (Keras needs this to
                                            materialize layer
                                            weights before load)
        3. model.load_weights(path)        (the trained weights
                                            from disk)
    The loader (app/services/translation_loader.py) does steps 2-3.
    This file only owns step 1.

WHY THE ARCHITECTURE LIVES IN CODE:
    Keras's .h5 weights file stores per-LAYER weight tensors keyed
    by layer name (e.g., "transformer/encoder_layers/0/self_attn/W_q/kernel").
    It does NOT contain the architectural graph. The deployment
    service must instantiate the same class hierarchy with the same
    layer names so load_weights can find a 1:1 mapping. Mismatched
    architecture -> Keras either raises a name-mismatch warning or
    silently leaves layers at random init.

ARCHITECTURE OVERVIEW:
    Encoder-decoder Transformer (Vaswani et al. 2017 recipe):

        src tokens (B, S_src)                tgt tokens (B, S_tgt)
              |                                    |
              v                                    v
        src_embedding * sqrt(d_model)        tgt_embedding * sqrt(d_model)
              |                                    |
              v                                    v
        positional encoding                  positional encoding
              |                                    |
              v                                    v
       [N_ENC_LAYERS encoder layers]   [N_DEC_LAYERS decoder layers]
              |                                    |
              v                                    v
              |  ----------------- cross attn ---->|
              |                                    |
                                                   v
                                            output_proj (Dense -> tgt_vocab)
                                                   |
                                                   v
                                            logits (B, S_tgt, vocab)

    Each encoder layer: self-attn + LayerNorm + FFN + LayerNorm
    (post-norm, dropout on residual connections).

    Each decoder layer: self-attn (causal mask) + LayerNorm +
    cross-attn (attends to encoder output) + LayerNorm + FFN +
    LayerNorm.

    Inference uses greedy autoregressive decoding driven from the
    router: encode the source once, then loop calling decode() with
    the partial target to extend by one token at a time. See the
    router for the loop.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app.schemas.translation import TRANSLATION_MAX_LENGTH

# Module constants
# ---------------------------------------------------------------------------
"""
Pulled from the modeling phase: TensorFlow/16-transformers/pipeline.ipynb
config cell + data/processed/transformers_translation/preprocessing_info.json.

These are the EXACT values used to train the registered weights;
changing any of them produces a different state_dict layout that
load_weights cannot map onto the .h5 contents.
"""

VOCAB_SIZE: int = 8000        # shared EN+ES BPE vocab
D_MODEL: int = 256            # embedding + attention dim
N_HEADS: int = 8              # multi-head attention
D_FF: int = 1024              # feed-forward inner dim (4x d_model, standard)
N_ENC_LAYERS: int = 3         # encoder stack depth
N_DEC_LAYERS: int = 3         # decoder stack depth
DROPOUT: float = 0.15         # training-time dropout rate; no-op at eval

# Sequence length cap. Shared with the schema so request validation,
# input truncation, and decoding all use the same number.
MAX_LEN: int = TRANSLATION_MAX_LENGTH

# Special token IDs from preprocessing_info.json. The decoder loop
# starts with BOS and stops on EOS; PAD is masked everywhere.
PAD_IDX: int = 0
BOS_IDX: int = 1
EOS_IDX: int = 2
UNK_IDX: int = 3


# Mask helpers
# ---------------------------------------------------------------------------


def create_pad_mask(seq: tf.Tensor, pad_idx: int) -> tf.Tensor:
    """
    Build a padding mask for attention.

    Args:
        seq: Token-ID tensor of shape (batch, seq_len).
        pad_idx: The pad token ID. Positions equal to this are masked.

    Returns:
        Mask tensor shape (batch, 1, 1, seq_len). 1.0 where seq is
        pad (will be subtracted from attention scores via -1e9 in
        MultiHeadAttention.call), 0.0 elsewhere.

    The two singleton dims are for broadcasting against the
    (batch, n_heads, q_len, k_len) attention-score tensor inside
    MultiHeadAttention. Broadcast rules align the last dim with
    k_len so a single key column for each padded position gets
    masked.
    """
    mask = tf.cast(tf.equal(seq, pad_idx), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_causal_mask(seq_len: int | tf.Tensor) -> tf.Tensor:
    """
    Build a lower-triangular causal mask for decoder self-attention.

    Args:
        seq_len: Length of the target sequence. Accepts Python int
            for static shapes or a scalar tf.Tensor (e.g., from
            tf.shape(x)[1]) for dynamic shapes - tf.ones and
            tf.linalg.band_part both accept either.

    Returns:
        Mask tensor shape (1, 1, seq_len, seq_len). 1.0 above the
        diagonal (future positions, to be masked), 0.0 on or below.

    band_part(input, num_lower=-1, num_upper=0) keeps the lower
    triangle including the diagonal; subtracting from 1.0 inverts
    to mask the strictly-upper triangle. The two singleton dims
    broadcast against (batch, n_heads, q_len, k_len) attention.
    """
    mask = 1.0 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]


# Building-block layers
# ---------------------------------------------------------------------------


class PositionalEncoding(keras.layers.Layer):
    """
    Sinusoidal positional encoding (Vaswani 2017).

    The position table is pre-computed at __init__ time as a numpy
    array, then stored as a tf.constant - no learnable parameters,
    no gradient flow. At call() time we just slice the appropriate
    prefix and add to the input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.dropout = keras.layers.Dropout(dropout_rate)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        # Add a leading batch dim so the constant broadcasts cleanly
        # against the (batch, seq_len, d_model) input tensor.
        self.pe = tf.constant(pe[np.newaxis, :, :])

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Slice PE to the input's actual sequence length, add, dropout.
        x = x + self.pe[:, : tf.shape(x)[1], :]
        return self.dropout(x, training=training)


class MultiHeadAttention(keras.layers.Layer):
    """
    Standard multi-head scaled dot-product attention.

    Four learned projections (W_q, W_k, W_v, W_o), all square
    Dense(d_model). The split into n_heads happens via reshape,
    not separate weight tensors - all heads share one parameter
    block per projection, just sliced into n_heads chunks during
    the forward pass.
    """

    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = keras.layers.Dense(d_model)
        self.W_k = keras.layers.Dense(d_model)
        self.W_v = keras.layers.Dense(d_model)
        self.W_o = keras.layers.Dense(d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(query)[0]
        # (B, S, d_model) -> (B, S, n_heads, d_k) -> (B, n_heads, S, d_k).
        # Transpose puts heads ahead of seq so matmul is over the last
        # two dims (S, d_k) for each head independently.
        q = tf.transpose(
            tf.reshape(self.W_q(query), (batch_size, -1, self.n_heads, self.d_k)),
            [0, 2, 1, 3],
        )
        k = tf.transpose(
            tf.reshape(self.W_k(key), (batch_size, -1, self.n_heads, self.d_k)),
            [0, 2, 1, 3],
        )
        v = tf.transpose(
            tf.reshape(self.W_v(value), (batch_size, -1, self.n_heads, self.d_k)),
            [0, 2, 1, 3],
        )
        # Scaled dot-product. transpose_b=True treats the second arg's
        # last two dims as transposed, so matmul gives (B, n_heads, S_q, S_k).
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.d_k, tf.float32)
        )
        if mask is not None:
            # Mask is 1.0 where attention should be blocked; -1e9
            # makes the softmax output ~0 there.
            scores += mask * -1e9
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        # Aggregate values + un-split heads.
        context = tf.matmul(attn_weights, v)
        context = tf.reshape(
            tf.transpose(context, [0, 2, 1, 3]), (batch_size, -1, self.d_model)
        )
        return self.W_o(context), attn_weights


class FeedForward(keras.layers.Layer):
    """
    Position-wise feed-forward network: Dense(d_ff, relu) + Dense(d_model).

    d_ff is typically 4x d_model in the Transformer recipe. Dropout
    after the first activation. No residual connection inside the
    layer - the encoder/decoder layer wrappers add that.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.linear1 = keras.layers.Dense(d_ff, activation="relu")
        self.linear2 = keras.layers.Dense(d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.linear2(self.dropout(self.linear1(x), training=training))


# Encoder + decoder layers
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(keras.layers.Layer):
    """
    One encoder layer: self-attn + residual+norm, FFN + residual+norm.

    Uses POST-norm (LayerNorm after the residual addition) which is
    the original Vaswani recipe. Modern variants often use pre-norm
    for training stability, but post-norm is what was trained here.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout_rate)
        self.drop2 = keras.layers.Dropout(dropout_rate)

    def call(
        self, src: tf.Tensor, src_mask: tf.Tensor | None = None, training: bool = False
    ) -> tf.Tensor:
        attn_out, _ = self.self_attn(src, src, src, mask=src_mask, training=training)
        src = self.norm1(src + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(src, training=training)
        src = self.norm2(src + self.drop2(ffn_out, training=training))
        return src


class TransformerDecoderLayer(keras.layers.Layer):
    """
    One decoder layer: self-attn + cross-attn + FFN, each with residual+norm.

    Three sub-blocks with three independent norms and three dropouts.
    The self-attention uses a causal+pad mask; the cross-attention
    queries the encoder output and uses just a pad mask on the
    encoder side.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout_rate)
        self.drop2 = keras.layers.Dropout(dropout_rate)
        self.drop3 = keras.layers.Dropout(dropout_rate)

    def call(
        self,
        tgt: tf.Tensor,
        encoder_out: tf.Tensor,
        src_mask: tf.Tensor | None = None,
        tgt_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        # 1. Self-attn: target attends to its own preceding tokens
        # (tgt_mask combines pad mask + causal triangle).
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask, training=training)
        tgt = self.norm1(tgt + self.drop1(self_attn_out, training=training))
        # 2. Cross-attn: target queries against encoder output.
        cross_attn_out, _ = self.cross_attn(
            tgt, encoder_out, encoder_out, mask=src_mask, training=training
        )
        tgt = self.norm2(tgt + self.drop2(cross_attn_out, training=training))
        # 3. Feed-forward.
        ffn_out = self.ffn(tgt, training=training)
        tgt = self.norm3(tgt + self.drop3(ffn_out, training=training))
        return tgt


# Full model
# ---------------------------------------------------------------------------


class Transformer(keras.Model):
    """
    Encoder-decoder Transformer for sequence-to-sequence translation.

    Public methods:
        encode(src, src_mask, training):
            Run the encoder stack once. Used by the router to cache
            encoder output during autoregressive decoding (avoids
            re-running the encoder for every generated token).
        decode(tgt, encoder_out, src_mask, tgt_mask, training):
            Run the decoder stack + output projection. Returns
            logits over the target vocabulary.
        call(src, tgt, training):
            Training-time end-to-end forward pass. Builds masks
            internally and runs encoder + decoder in one shot.
            Used by Keras during model.fit() and by the loader's
            warm-up pass; the inference router uses encode + decode
            separately for efficiency.

    The src_embedding/tgt_embedding outputs are scaled by
    sqrt(d_model) before adding positional encoding. This
    standardizes the magnitude of token embeddings against the PE,
    which has fixed per-component variance regardless of d_model.
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int,
        n_heads: int,
        n_enc_layers: int,
        n_dec_layers: int,
        d_ff: int,
        max_len: int,
        dropout_rate: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.scale = tf.math.sqrt(tf.cast(d_model, tf.float32))
        self.src_embedding = keras.layers.Embedding(src_vocab, d_model)
        self.tgt_embedding = keras.layers.Embedding(tgt_vocab, d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len, dropout_rate=dropout_rate)
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_enc_layers)
        ]
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_dec_layers)
        ]
        self.output_proj = keras.layers.Dense(tgt_vocab)

    def encode(
        self, src: tf.Tensor, src_mask: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        x = self.src_embedding(src) * self.scale
        x = self.pe(x, training=training)
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask, training=training)
        return x

    def decode(
        self,
        tgt: tf.Tensor,
        encoder_out: tf.Tensor,
        src_mask: tf.Tensor,
        tgt_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        x = self.tgt_embedding(tgt) * self.scale
        x = self.pe(x, training=training)
        for layer in self.decoder_layers:
            x = layer(
                x, encoder_out, src_mask=src_mask, tgt_mask=tgt_mask, training=training
            )
        return self.output_proj(x)

    def call(self, src: tf.Tensor, tgt: tf.Tensor, training: bool = False) -> tf.Tensor:
        src_mask = create_pad_mask(src, self.pad_idx)
        tgt_pad_mask = create_pad_mask(tgt, self.pad_idx)
        tgt_causal = create_causal_mask(tf.shape(tgt)[1])
        # Combine pad and causal masks: any position that's EITHER
        # pad OR future is masked. tf.maximum is the elementwise OR
        # for {0.0, 1.0}-valued masks.
        tgt_mask = tf.maximum(tgt_pad_mask, tgt_causal)
        encoder_out = self.encode(src, src_mask, training=training)
        return self.decode(tgt, encoder_out, src_mask, tgt_mask, training=training)
