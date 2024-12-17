import torch
from torch.utils.data import Dataset

from preprocessing.vocab.liar_vocab import build_vocab_from_series


class LiarDataset(Dataset):

    def __init__(self, df, transform=None, target_transform=None):
        # data
        self.df = df
        # Textual data
        self.label = df["label"]
        self.statement = df["statement"]
        self.subject = df["subject"]
        self.speaker = df["speaker"]
        self.speaker_job_title = df["job_title"]
        self.state_info = df["state_info"]
        self.party_affiliation = df["party_affiliation"]
        self.context_venue_or_location = df["context_venue_or_location"]
        # numerical data
        self.id = df["id"]
        self.barely_true_count = df["barely_true_counts"]
        self.false_count = df["false_counts"]
        self.half_true_count = df["half_true_counts"]
        self.mostly_true_count = df["mostly_true_counts"]
        self.pants_on_fire_count = df["pants_on_fire_counts"]

        # Create vocab for each text column
        self.statement_vocab = build_vocab_from_series(self.statement)
        self.label_vocab = build_vocab_from_series(self.label)
        self.subjects_vocab = build_vocab_from_series(self.subjects_vocab)
        self.speaker_vocab = build_vocab_from_series(self.speaker_vocab)
        self.speaker_job_title_vocab = build_vocab_from_series(
            self.speaker_job_title_vocab
        )
        self.state_info_vocab = build_vocab_from_series(self.state_info_vocab)
        self.party_affiliation_vocab = build_vocab_from_series(
            self.party_affiliation_vocab
        )
        self.context_venue_or_location_vocab = build_vocab_from_series(
            self.context_venue_or_location_vocab
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.label[idx]

        statement = self.statement[idx]
        subject = self.subject[idx]
        speaker = self.speaker[idx]
        speaker_job_title = self.speaker_job_title[idx]
        state_info = self.state_info[idx]
        party_affiliation = self.party_affiliation[idx]
        context_venue_or_location = self.context_venue_or_location[idx]

        statement_id = self.id[idx]
        barely_true_count = self.barely_true_count[idx]
        false_count = self.false_count[idx]
        half_true_count = self.half_true_count[idx]
        mostly_true_count = self.mostly_true_count[idx]
        pants_on_fire_count = self.pants_on_fire_count[idx]

        statement_tokens = self._tokenize(statement, self.statement_vocab)
        subject_tokens = self._tokenize(subject, self.subjects_vocab)
        speaker_tokens = self._tokenize(speaker, self.speaker_vocab)
        speaker_job_title_tokens = self._tokenize(
            speaker_job_title, self.speaker_job_title_vocab
        )
        state_info_tokens = self._tokenize(state_info, self.state_info_vocab)
        party_affiliation_tokens = self._tokenize(
            party_affiliation, self.party_affiliation_vocab
        )
        context_venue_or_location_tokens = self._tokenize(
            context_venue_or_location, self.context_venue_or_location_vocab
        )

        return {
            "id": statement_id,
            "barely_true_count": barely_true_count,
            "false_count": false_count,
            "half_true_count": half_true_count,
            "mostly_true_count": mostly_true_count,
            "pants_on_fire_count": pants_on_fire_count,
            "statement": torch.tensor(statement_tokens),
            "subject": torch.tensor(subject_tokens),
            "speaker": torch.tensor(speaker_tokens),
            "speaker_job_title": torch.tensor(speaker_job_title_tokens),
            "state_info": torch.tensor(state_info_tokens),
            "party_affiliation": torch.tensor(party_affiliation_tokens),
            "context_venue_or_location": torch.tensor(context_venue_or_location_tokens),
            "label": label,
        }

    def _tokenize(self, element, vocab):
        return [vocab[token] for token in element.split()]
