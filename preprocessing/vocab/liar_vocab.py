from torchtext.vocab import build_vocab_from_iterator

from preprocessing import preprocess_text
from preprocessing.liar.preprocess_liar import preprocess_liar_statements
from preprocessing.preprocessing_config import PreprocessingConfig


def yield_tokens(series):
    for text in series:
        yield text.split()


def build_vocab_from_series(
    series, config: PreprocessingConfig = PreprocessingConfig()
):
    processed_text = preprocess_text(series, config)
    liar_vocab = build_vocab_from_iterator(
        yield_tokens(processed_text), specials=["<unk>", "<pad>"]
    )
    liar_vocab.set_default_index(liar_vocab["<unk>"])
    return liar_vocab


# df = preprocess_liar_statements(train_df)
# c
# liar_vocab.set_default_index(liar_vocab['<unk>'])
