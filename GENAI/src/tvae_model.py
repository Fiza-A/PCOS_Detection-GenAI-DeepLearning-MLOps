"""Train TVAE on the deterministic training partition only."""
from generator_common import train_and_sample


def run():
    return train_and_sample("tvae")
