from torch.nn.utils.rnn import pad_sequence


# We do the padding here because the sentences in each batch should have the same dimension.
class MyCollate:
    # pad_idx is the index for the <pad> token
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # {
        #     "id": statement_id,
        #     "barely_true_count": barely_true_count,
        #     "false_count": false_count,
        #     "half_true_count": half_true_count,
        #     "mostly_true_count": mostly_true_count,
        #     "pants_on_fire_count": pants_on_fire_count,
        #     "statement": statement_tokens,
        #     "subject": subject_tokens,
        #     "speaker_": speaker_tokens,
        #     "speaker_job_title": speaker_job_title_tokens,
        #     "label": label,
        #     "state_info": state_info_tokens,
        #     "party_affiliation": party_affiliation_tokens,
        #     "context_venue_or_location": context_venue_or_location_tokens,
        # }

        # Batch is a tuple of sentences and their respective labels, this can be changed depending on the input fields we are using for training
        # by changing the get item function in the dataset class.
        barely_true_counts = [x["barely_true_count"] for x in batch]
        false_counts = [x["false_count"] for x in batch]
        half_true_counts = [x["half_true_count"] for x in batch]
        mostly_true_counts = [x["mostly_true_count"] for x in batch]
        pants_on_fire_counts = [x["pants_on_fire_count"] for x in batch]
        statements = [x["statement"] for x in batch]
        subjects = [x["subject"] for x in batch]
        speakers = [x["speaker_"] for x in batch]
        speaker_job_titles = [x["speaker_job_title"] for x in batch]
        labels = [x["label"] for x in batch]
        state_infos = [x["state_info"] for x in batch]
        party_affiliations = [x["party_affiliation"] for x in batch]
        context_venue_or_locations = [x["context_venue_or_location"] for x in batch]
        paded_sentences = pad_sequence(
            sentences, batch_first=True, padding_value=self.pad_idx
        )

        return paded_sentences, torch.tensor(labels)
