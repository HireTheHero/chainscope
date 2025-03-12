# ChainScope

This repository contains the datasets and evaluation scripts for the [Chain-of-Thought Reasoning In The Wild Is Not Always Faithful](https://arxiv.org/abs/2503.08679) paper.

## Setup Instructions

1. install python3.12
1. setup your private SSH key
   1. put it under in `.ssh/id_[protocol]`
   1. `chmod 600 [key]`
   1. you can debug with `ssh -T -v git@github.com`
1. clone the repo via ssh `git@github.com:jettjaniak/chainscope.git`
1. make virtual env `python3.12 -m venv .venv`
1. activate virtual env `source .venv/bin/activate`
1. upgrade pip `pip install --upgrade pip`
1. install project in editable state `pip install -e .`
1. install pre-commit hooks `pre-commit install && pre-commit run`
1. run `pytest`

# Datasets

Comparative questions are based on a small database of properties of different types of objects in `chainscope/data/properties` and generated programatically by `scripts/gen_qs.py`.

All the datsets for IPHR can be found in `chainscope/data/questions`. The ones relevant for the paper are the yamls starting with "wm".

The datasets for Restoration Errors can be found in `chainscope/data/problems`.

# Implicit Post-Hoc Rationalization

We generate CoT responses for the models using `scripts/gen_cots.py`, and evaluate them using `scripts/eval_cots.py`.

# Restoration Errors and Unfaithful Shortcuts

The repo also includes experiments for evaluating Restoration Errors and Unfaithful Shortcuts (see [our arXiv paper](https://arxiv.org/abs/2503.08679) for definitions, etc) -- exact scripts and data from our paper are **currently being ported** -- sorry for the wait!

## `putnamlike` scripts

To evaluate Restoration Errors and Unfaithful Shortcuts, we use the `putnamlike` pipeline of scripts.

1. `putnamlike0_save_rollouts.py`
2. `putnamlike1_are_rollouts_correct.py`
3. `putnamlike2_split_cots.py`
   * WARNING! This is somewhat unreliable, particularly for really long rollouts, as it does only very basic checks of the correct format by checking that the length of the steps added together is within 10% of the original response length. Empirically, using Claude 3.7 Sonnet is much better than other LLMs (far more max output tokens, and not lazy)
4. `putnamlike2p5_critical_steps_eval.py`
   * Optional, reduces number of steps to evaluate
   * If used, then also pass the flag `--critical_steps_yaml=...` to `putnamlike3_main_faithfulness_eval.py`
5. `putnamlike3_main_faithfulness_eval.py`
   * Pass the file output from `putnamlike2_split_cots.py` to this (and possibly `--critical_steps_yaml=...` too, see `putnamlike2p5_critical_steps_eval.py`)

# Citation

To cite this work, you can use [our arXiv paper](https://arxiv.org/abs/2503.08679) citation:

```
@misc{arcuschin2025chainofthoughtreasoningwildfaithful,
      title={Chain-of-Thought Reasoning In The Wild Is Not Always Faithful}, 
      author={Iván Arcuschin and Jett Janiak and Robert Krzyzanowski and Senthooran Rajamanoharan and Neel Nanda and Arthur Conmy},
      year={2025},
      eprint={2503.08679},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.08679}, 
}
```
