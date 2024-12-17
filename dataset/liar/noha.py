import os
import zipfile
import requests
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ---- Utility Function to Download and Extract Dataset ----
def download_and_extract_dataset(url, output_dir="dataset"):
    """
    Downloads a dataset from the given URL, extracts its contents, and organizes them.

    Parameters:
        url (str): URL to the dataset.
        output_dir (str): Directory to save the dataset.

    Returns:
        str: Path to the extracted dataset directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "dataset.zip")

    # Download the dataset
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    # Extract the dataset
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)
    print(f"Dataset downloaded and extracted to '{output_dir}'.")
    return output_dir


# ---- Load Dataset into Pandas DataFrame ----
def load_dataset(file_path, column_names):
    """
    Loads a TSV file into a Pandas DataFrame with specified column names.

    Parameters:
        file_path (str): Path to the TSV file.
        column_names (list): List of column names for the DataFrame.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path, delimiter="\t", header=None, names=column_names)


# ---- LiarDataset Class ----
class LiarMLDataset(Dataset):
    """
    A PyTorch Dataset for the LIAR dataset.

    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset.
        vocab (dict): Vocabulary mapping words to indices.
        transform (callable, optional): Optional transform to apply to inputs.
        target_transform (callable, optional): Optional transform to apply to targets.
    """

    def __init__(self, df, vocab, transform=None, target_transform=None):
        self.id = df["ID"]

        self.label = df["Label"]
        self.statement = df["Statement"]
        self.subjects = df["Subjects"]
        self.speaker = df["Speaker"]
        self.speaker_job_title = df["Speaker_Job_Title"]
        self.state_info = df["State_Info"]
        self.party_affiliation = df["Party_Affiliation"]
        self.barely_true_count = df["Barely_True_Count"]
        self.false_count = df["False_Count"]
        self.half_true_count = df["Half_True_Count"]
        self.mostly_true_count = df["Mostly_True_Count"]
        self.pants_on_fire_count = df["Pants_On_Fire_Count"]
        self.context = df["Context"]
        self.vocab = vocab
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.statement)

    def __getitem__(self, idx):
        example_id = self.id.iloc[idx]
        label = self.label.iloc[idx]
        statement = self.statement.iloc[idx]
        subjects = self.subjects.iloc[idx]
        speaker = self.speaker.iloc[idx]
        speaker_job_title = self.speaker_job_title.iloc[idx]
        state_info = self.state_info.iloc[idx]
        party_affiliation = self.party_affiliation.iloc[idx]
        barely_true_count = self.barely_true_count.iloc[idx]
        false_count = self.false_count.iloc[idx]
        half_true_count = self.half_true_count.iloc[idx]
        mostly_true_count = self.mostly_true_count.iloc[idx]
        pants_on_fire_count = self.pants_on_fire_count.iloc[idx]
        context = self.context.iloc[idx]

        # Convert tokens to numerical tokens using the vocab
        numerical_tokens = [
            self.vocab.get(token, self.vocab["<UNK>"]) for token in statement.split()
        ]
        numerical_tokens = torch.tensor(numerical_tokens, dtype=torch.long)

        return numerical_tokens, label


# ---- MyCollate Class ----
class MyCollate:
    """
    A custom collation function for padding variable-length sequences in batches.

    Parameters:
        pad_idx (int): Index to use for padding.
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Pads sequences and collates them into a batch.

        Parameters:
            batch (list): List of (numerical_tokens, label) tuples.

        Returns:
            tuple: Padded sequences and tensor of labels.
        """
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        # Pad sequences to the same length
        padded_sentences = pad_sequence(
            sentences, batch_first=True, padding_value=self.pad_idx
        )

        return padded_sentences, torch.tensor(labels, dtype=torch.long)
