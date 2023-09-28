# %%
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import expm  # type: ignore
from enum import Enum


# %%
class Allele(Enum):
    A = 0
    C = 1
    G = 2
    T = 3


class Genotype:
    def __init__(self, first: Allele, second: Allele):
        self.first = first
        self.second = second

    def flatten(self) -> int:
        return self.first.value * len(Allele) + self.second.value

    @staticmethod
    def enumerate():
        return [Genotype(first, second) for first in Allele for second in Allele]

    def __eq__(self, other: object):
        if isinstance(other, Genotype):
            return self.first == other.first and self.second == other.second
        return False


num_states = len(Genotype.enumerate())

epsilon = 0
delta = 0


def genotype_likelihood(observed: Genotype, actual: Genotype) -> float:
    """P(observed | actual)"""

    observed_is_homo = observed.first == observed.second
    actual_is_homo = actual.first == actual.second

    first_matches = observed.first == actual.first
    second_matches = observed.second == actual.second

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

    prior = 1 / num_states  # P(actual); assume same for all genotypes
    likelihood = genotype_likelihood(observed, actual)

    evidence = 0

    for a1 in Allele:
        for a2 in Allele:
            evidence += genotype_likelihood(observed, Genotype(a1, a2)) * prior

    return likelihood * prior / evidence


# %%
Q = np.eye(num_states)


class Node(ABC):
    @abstractmethod
    def likelihood(self, state: Genotype) -> float:
        pass


class Leaf(Node):
    def __init__(self, state: Genotype):
        self.state = state

    def likelihood(self, state: Genotype) -> float:
        return 1 if state == self.state else 0


class Parent(Node):
    def __init__(self, tL: float, nodeL: Node, tR: float, nodeR: Node):
        self.tL = tL
        self.nodeL = nodeL
        self.tR = tR
        self.nodeR = nodeR

    @staticmethod
    def __pr(cur_state: Genotype, t: float):
        cur_state_one_hot = np.zeros(num_states)
        cur_state_one_hot[cur_state.flatten()] = 1

        return expm(Q * t) @ cur_state_one_hot

    def likelihood(self, state: Genotype) -> float:
        likelihood_arr_L = np.array(
            [self.nodeL.likelihood(g) for g in Genotype.enumerate()]
        )
        likelihood_arr_R = np.array(
            [self.nodeR.likelihood(g) for g in Genotype.enumerate()]
        )

        totalL = np.dot(Parent.__pr(state, self.tL), likelihood_arr_L)
        totalR = np.dot(Parent.__pr(state, self.tR), likelihood_arr_R)

        return totalL * totalR

    def treeLikelihood(self) -> float:
        prior = 1 / num_states
        return sum([prior * self.likelihood(g) for g in Genotype.enumerate()])


# %%

# for brevity
A = Allele
G = Genotype

# parent1 = Parent(1, Leaf(G(A.A, A.A)), 1, Leaf(G(A.A, A.A)))
# parent2 = Parent(0.5, Leaf(G(A.A, A.A)), 0.5, Leaf(G(A.A, A.A)))
# parent3 = Parent(0.5, parent1, 1.5, Leaf(G(A.A, A.A)))
# parent4 = Parent(1, parent3, 2, parent2)
# parent5 = Parent(0.5, parent4, 2.5, Leaf(G(A.A, A.A)))

# print(parent5.treeLikelihood())

parent1 = Parent(1, Leaf(G(A.A, A.A)), 1, Leaf(G(A.A, A.A)))
print(parent1.treeLikelihood())
