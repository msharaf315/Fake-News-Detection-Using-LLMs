# Liar constants
LIAR_LABELS_TO_INDEX = {
    "pants-fire": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "false": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "barely-true": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "half-true": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "mostly-true": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "true": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}

LIAR_HEADER = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context_venue_or_location",
]

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
