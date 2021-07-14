from typing import Sequence

import flax.linen as nn


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


def assertEqual(actual, expected, msg, first="Got", second="Expected"):
    if actual != expected:
        raise ValueError(msg + f' {first}: "{actual}" {second}: "{expected}"')


def assertIn(actual, expected, msg, first="Got", second="Expected one of"):
    if actual not in expected:
        raise ValueError(msg + f' {first}: "{actual}" {second}: {expected}')
