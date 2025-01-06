from torchtext.vocab import build_vocab_from_iterator

from constants.liar_constants import PAD_TOKEN, UNKNOWN_TOKEN


def yield_tokens(series):
    for text in series:
        yield text.split()


def build_vocab_from_series(series):

    liar_vocab = build_vocab_from_iterator(
        yield_tokens(series), specials=[UNKNOWN_TOKEN, PAD_TOKEN]
    )
    liar_vocab.set_default_index(liar_vocab[UNKNOWN_TOKEN])
    return liar_vocab
