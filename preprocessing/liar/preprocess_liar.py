from constants.liar_constants import LIAR_LABELS_TO_INDEX
from preprocessing import preprocess_text
from preprocessing.preprocessing_config import PreprocessingConfig


def preprocess_liar_statements(df):
    config = PreprocessingConfig()
    df["processed_statement"] = preprocess_text(df["statement"], config)
    df["labels_index"] = df["label"].apply(lambda x: LIAR_LABELS_TO_INDEX[x])
    return df
