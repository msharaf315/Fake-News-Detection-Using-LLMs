import string
from preprocessing.preprocessing_config import PreprocessingConfig
from nltk.corpus import stopwords
import nltk
import re


def preprocess_text(df_series, config: PreprocessingConfig):
    if config.remove_quotations:
        df_series = df_series.apply(lambda x: re.sub("'", "", x))

    if config.remove_punctuation:
        punctuation = set(string.punctuation)
        df_series = df_series.apply(
            lambda x: "".join(ch for ch in x if ch not in punctuation)
        )

    if config.remove_stop_words:
        nltk.download("stopwords")
        stop_words = stopwords.words("english")
        df_series = df_series.apply(
            lambda x: " ".join([word for word in x.split() if word not in (stop_words)])
        )

    if config.to_lower_case:
        df_series = df_series.apply(lambda x: x.lower())

    if config.add_sos_eos_tokens:
        df_series = df_series.apply(lambda x: "<sos> " + x + " <eos>")

    return df_series
