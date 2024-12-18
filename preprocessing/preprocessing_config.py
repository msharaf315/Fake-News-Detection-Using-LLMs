class PreprocessingConfig:

    def __init__(
        self,
        remove_quotations: bool = False,
        remove_punctuation: bool = False,
        remove_stop_words: bool = False,
        to_lower_case: bool = False,
        fill_na: bool = True,
        add_sos_eos_tokens: bool = True,
    ):
        self.remove_quotations = remove_quotations
        self.remove_punctuation = remove_punctuation
        self.remove_stop_words = remove_stop_words
        self.to_lower_case = to_lower_case
        self.add_sos_eos_tokens = add_sos_eos_tokens
        self.fill_na = fill_na
