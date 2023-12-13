from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
import numpy as np
from .item import Item
import ray
import torch

# For ANSI escape codes, see section 'Colors' (especially '24-bit') on page:
# https://en.wikipedia.org/wiki/ANSI_escape_code


def fgcol(r: int, g: int, b: int) -> str:
    """Sets the foreground color using a ANSI sequence. See https://en.wikipedia.org/wiki/ANSI_escape_code """
    return f'\x1b[38;2;{r};{g};{b}m'


def bgcol(r: int, g: int, b: int) -> str:
    """Sets the background color using a ANSI sequence. See https://en.wikipedia.org/wiki/ANSI_escape_code """
    return f'\x1b[48;2;{r};{g};{b}m'


def ansi_reset() -> str:
    """Resets the console color using a ANSI sequence. See https://en.wikipedia.org/wiki/ANSI_escape_code """
    return '\x1b[39;49m'


def print_color(text: str, bgcolor: list = [255, 255, 255], fgcolor: list = [0, 0, 0], **kwargs):
    """Prints the given text with the given colors."""
    print(f'{bgcol(*bgcolor)}{fgcol(*fgcolor)}{text}{ansi_reset()}', **kwargs)


def print_item(heightmap: np.ndarray, item: Item):
    """Prints the given Item in the given heightmap."""
    item_mask2d = np.zeros_like(heightmap)
    item_mask2d[item.min(0):item.max(0), item.min(1):item.max(1)] = 1
    for i in np.arange(heightmap.shape[0]):
        for j in np.arange(heightmap.shape[1]):
            color = [150, 255, 150] if item_mask2d[i, j] else [255, 255, 255]
            print_color('{:^3}'.format(heightmap[i, j]), color, end='')
        print()  # linebreak after row


def get_best_trial(analysis: ExperimentAnalysis, metric: str, mode: str = 'max') -> str:
    """Returns the best trial of the given analysis for the given metric."""
    assert mode in ['min', 'max']
    if mode == 'max':
        mode = np.max
        argmode = np.argmax
    elif mode == 'min':
        mode = np.min
        argmode = np.argmin
    trials = list(analysis.trial_dataframes.keys())
    trials_metrics = [mode(analysis.trial_dataframes[i][metric]) for i in trials]
    return trials[argmode(trials_metrics)]


def get_action_probabilities(info, space):
    """Computes the action probablities contained in the given info in the given space."""
    # actions, state_outs, info = policy.compute_single_action(pp.transform(observation))
    logits = info['action_dist_inputs']
    logits_dict = ray.rllib.models.modelv2.restore_original_dimensions(torch.from_numpy(logits).reshape(1, -1), space,
                                                                       tensorlib='torch')
    action_probs = {}
    for action, dist in logits_dict.items():
        action_probs[action] = ray.rllib.utils.numpy.softmax(dist.numpy().flatten()).reshape(dist.shape[1:])
    return action_probs
