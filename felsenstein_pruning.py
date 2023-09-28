# %%
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import expm
from enum import Enum

# %%
epsilon = 0.05
delta = 0.1


class Allele(Enum):
    A = 0
    C = 1
    G = 2
    T = 3


Genotype = tuple[Allele, Allele]


def genotype_likelihood(observed: Genotype, actual: Genotype) -> float:
    """P(observed | actual)"""

    observed_is_homo = observed[0] == observed[1]
    actual_is_homo = actual[0] == actual[1]

    first_matches = observed[0] == actual[0]
    second_matches = observed[1] == actual[1]

    if actual_is_homo:
        if first_matches and second_matches:  # aa|aa
            return 1 - epsilon + (1 / 2) * delta * epsilon
        elif first_matches or second_matches:  # ab|aa or ba|aa
            return (1 - delta) * (1 / 6) * epsilon
        elif observed_is_homo:  # bb | aa
            return (1 / 6) * delta * epsilon
        else:
            return 0
    else:
        if observed_is_homo and (first_matches or second_matches):  # aa|ab
            return (1 / 2) * delta + (1 / 6) * epsilon - (1 / 3) * delta * epsilon
        elif observed_is_homo:  # cc|ab
            return (1 / 6) * delta * epsilon
        elif first_matches and second_matches:  # ab|ab
            return (1 - delta) * (1 - epsilon)
        elif first_matches or second_matches:  # ac|ab
            return (1 - delta) * (1 / 6) * epsilon
        else:
            return 0


def genotype_posterior(actual: Genotype, observed: Genotype) -> float:
    """P(actual | observed)"""

    prior = 1 / 16  # P(actual); assume same for all genotypes
    likelihood = genotype_likelihood(observed, actual)

    evidence = 0

    for a1 in Allele:
        for a2 in Allele:
            evidence += genotype_likelihood(observed, (a1, a2)) * prior

    return likelihood * prior / evidence


# %%
num_states = 3
Q = np.array([[-2, 1, 1], [1, -2, 1], [1, 1, -2]])


def pr(cur_state: int, t: float):
    cur_state_one_hot = np.zeros(num_states)
    cur_state_one_hot[cur_state] = 1

    return expm(Q * t) @ cur_state_one_hot


class Node(ABC):
    @abstractmethod
    def likelihood(self, state: int) -> float:
        pass


class Leaf(Node):
    def __init__(self, state: int):
        self.state = state

    def likelihood(self, state: int) -> float:
        return 1 if state == self.state else 0


class Parent(Node):
    def __init__(self, tL: float, nodeL: Node, tR: float, nodeR: Node):
        self.tL = tL
        self.nodeL = nodeL
        self.tR = tR
        self.nodeR = nodeR

    def likelihood(self, state: int) -> float:
        likelihood_arr_L = np.array(
            [self.nodeL.likelihood(i) for i in range(num_states)]
        )
        likelihood_arr_R = np.array(
            [self.nodeR.likelihood(i) for i in range(num_states)]
        )

        totalL = np.dot(pr(state, self.tL), likelihood_arr_L)
        totalR = np.dot(pr(state, self.tR), likelihood_arr_R)

        return totalL * totalR

    def treeLikelihood(
        self, prior: list[float] = [1 / num_states] * num_states
    ) -> float:
        return sum([prior[i] * self.likelihood(i) for i in range(num_states)])


# %%
parent1 = Parent(1, Leaf(0), 1, Leaf(1))
parent2 = Parent(0.5, Leaf(2), 0.5, Leaf(2))
parent3 = Parent(0.5, parent1, 1.5, Leaf(0))
parent4 = Parent(1, parent3, 2, parent2)
parent5 = Parent(0.5, parent4, 2.5, Leaf(1))

parent5.treeLikelihood()
