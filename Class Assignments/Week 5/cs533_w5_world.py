import seedbank
import numpy as np
import pandas as pd
from textwrap import dedent


class Experiment:
    """
    Code for simulating a fake world for the CS 533 Week 5 assignment.
    """
    rng: np.random.Generator
    

    def __init__(self, *, paired=None):
        self.rng = seedbank.numpy_rng(seedbank.root_seed())
        if paired is None:
            self.mode = self.rng.choice(['alpha', 'beta', 'gamma'])
        elif paired:
            self.mode = 'gamma'
        else:
            self.mode = 'alpha'
        self.mean = self.rng.normal(500, 10)
        self.diff = self.rng.normal(0, 10)
        self.varA = self.rng.gamma(100, .25)
        self.varB = self.rng.gamma(100, .25)


    def describe(self):
        """
        Describe the experiment.
        """
        if self.mode == 'alpha':
            return dedent('''
                This experiment measures the nonsense output of fabulators under different
                conditions.
                
                This experiment randomly assigns each subject (fabulator) one of two different
                treatments, and measures the amount of nonsense they produce in a 10-minute period
                under otherwise-identical conditions.

                Your goal is to measure assess whether condition A results in more (or less) nonsense
                than condition B.
            ''').strip()
        elif self.mode == 'beta':
            return dedent('''
                This experiment measures the nonsense output of fabulators trained at different
                schools to see if they produce nonsense at different rates.
                
                It measures how much nonsense each fabulator produces in a 10-minute period under
                identical conditions.

                Your goal is to measure assess whether school A's fabulators produce more (or less)
                nonsense than those from school B.
            ''').strip()
        elif self.mode == 'gamma':
            return dedent('''
                This experiment measures the nonsense output of fabulators under different
                conditions.  It measures each fabulator twice, under two different conditions,
                to see how much nonsense they produce in each condition.

                Your goal is to measure whether condition A causes fabulators to produce more
                (or less) nonsense than condition B.
            ''').strip()

    
    def run_experiment(self, n=100):
        """
        Run the experiment once with a specified sample size, and return the results.
        """
        baseline = self.rng.normal(0, 1, n)
        xA = baseline * np.sqrt(self.varA) + self.mean
        
        if self.mode == 'gamma':
            diff = self.rng.normal(self.diff, self.varB, n)
            xB = xA - diff
            return pd.DataFrame({
                'subject': np.arange(n) + 1,
                'CondA_Nonsense': xA,
                'CondB_Nonsense': xB
            })

        xB = self.rng.normal(self.mean - self.diff, self.varB, n)
        col = 'Condition' if self.mode == 'alpha' else 'School'
        return pd.concat([
            pd.DataFrame({
                'subject': np.arange(n) + 1,
                col: 'A',
                'Nonsense': xA
            }),
            pd.DataFrame({
                'subject': np.arange(n) + n + 1,
                col: 'B',
                'Nonsense': xB
            }),
        ], ignore_index=True)

    def answers(self):
        pu = 'paired' if self.mode == 'gamma' else 'unpaired'
        ef = self.diff
        return 'This was an {} experiment with effect size {:.4f}'.format(pu, ef)