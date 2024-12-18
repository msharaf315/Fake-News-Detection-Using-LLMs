from constants.liar_constants import LIAR_LABELS_TO_INDEX
from preprocessing.preprocess_text import preprocess_text
from preprocessing.preprocessing_config import PreprocessingConfig


def preprocess_liar_statements(df):
    config = PreprocessingConfig()
    df["statement"] = preprocess_text(df["statement"], config)
    df["subject"] = preprocess_text(df["subject"], config)
    df["speaker"] = preprocess_text(df["speaker"], config)
    df["job_title"] = preprocess_text(df["job_title"], config)
    df["state_info"] = preprocess_text(df["state_info"], config)
    df["party_affiliation"] = preprocess_text(df["party_affiliation"], config)
    df["context_venue_or_location"] = preprocess_text(
        df["context_venue_or_location"], config
    )
    df["labels_index"] = df["label"].apply(lambda x: LIAR_LABELS_TO_INDEX[x])
    return df
