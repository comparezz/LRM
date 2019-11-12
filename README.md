# Learning Reward Machines for Partially Observable Reinforcement Learning

Reward Machines (RMs), originally proposed for specifying problems in Reinforcement Learning (RL), provide a structured, automata-based representation of a reward function that allows an agent to decompose problems into subproblems that can be efficiently learned using off-policy learning. Here we show that RMs can be learned from experience, instead of being specified by the user, and that the resulting problem decomposition can be used to effectively solve partially observable RL problems. We pose the task of learning RMs as a discrete optimization problem where the objective is to find an RM that decomposes the problem into a set of subproblems such that the combination of their optimal memoryless policies is an optimal policy for the original problem. A detailed description of our approach and main results can be found in the following paper ([link](http://www.cs.toronto.edu/~rntoro/docs/LRM_paper.pdf)):

    @inproceedings{tor-etal-neurips19,
        author = {Toro Icarte, Rodrigo and Waldie, Ethan and Klassen, Toryn Q. and Valenzano, Richard and Castro, Margarita P. and McIlraith, Sheila A.},
        title     = {Learning Reward Machines for Partially Observable Reinforcement Learning},
        booktitle = {Proceedings of the 33rd Conference on Neural Information Processing Systems (NeurIPS)},
        year      = {2019}
    }

We are working towards releasing a clean (and usable) version of our code within the next few weeks. If you fill this form ([link](https://docs.google.com/forms/d/e/1FAIpQLSfKHJd9yyfx-2-p_tdM5fhSVfd5WK2vcsnIjMruV21MARI4jA/viewform?usp=sf_link)), we will let you know as soon as our code is released. Thank you for your patience :)
