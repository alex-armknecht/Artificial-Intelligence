'''
ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.

@author: <Emilie Coon; Makena Robison; Alex A>
'''
from pomegranate import *
import pandas as pd
import numpy 
import math
import itertools
import unittest


class AdEngine:

    def __init__(self, data_file, dec_vars, util_map):
        """
        Responsible for initializing the Decision Network of the
        AdEngine using the following inputs
        
        :param string data_file: path to csv file containing data on which
        the network's parameters are to be learned
        :param list dec_vars: list of string names of variables to be
        considered decision variables for the agent. Example:
          ["Ad1", "Ad2"]
        :param dict util_map: discrete, tabular, utility map whose keys
        are variables in network that are parents of a utility node, and
        values are dictionaries mapping that variable's values to a utility
        score, for example:
          {
            "X": {0: 20, 1: -10}
          }
        represents a utility node with single parent X whose value of 0
        has a utility score of 20, and value 1 has a utility score of -10
        """
        self.data_file = pd.read_csv(data_file)
        self.variable_names = list(self.data_file.columns)
        self.network = BayesianNetwork.from_samples(self.data_file, algorithm = 'exact', state_names = self.variable_names)
        self.util_map = util_map
        self.dec_vars = dec_vars

        

    def meu(self, evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, selects the ad content that maximizes expected utility
        and returns a dictionary over any decision variables and their
        best values plus the MEU from making this selection.
        
        :param dict evidence: dict mapping network variables to their
        observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: 2-Tuple of the format (a*, MEU) where:
          - a* = dict of format: {"DecVar1": val1, "DecVar2": val2, ...}
          - MEU = float representing the EU(a* | evidence)
        """
        best_decisions, best_util = dict(), -math.inf 
        query_letter = list(self.util_map.keys())[0]
        index_of_letter = list(self.data_file.columns).index(query_letter)
        all_actions = self.get_actions()
        best_util = 0
        if len(self.dec_vars) == 1:
            for action in all_actions:
                print(" -- " + str(action))
                dict_action = {}
                dict_action[self.dec_vars[0]] = action
                current_val = self.find_EU(dict_action, evidence, query_letter, index_of_letter)
                if current_val > best_util : 
                    best_util = current_val
                    best_decisions = {str(self.dec_vars[0]): action}
        else:
            for action in all_actions:
                print(action)
                dict_action = {}
                for a in range(len(action)):
                    dict_action[self.dec_vars[a]] = action[a]  
                current_val = self.find_EU(dict_action, evidence, query_letter, index_of_letter)
                if current_val > best_util : 
                    best_util = current_val
                    for d in range(len(self.dec_vars)):
                        best_decisions[self.dec_vars[d]] = action[d]     
        return (best_decisions, best_util)


    def vpi(self, potential_evidence, observed_evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.
        
        :param string potential_evidence: string representing the variable name
        of the variable under evaluation
        :param dict observed_evidence: dict mapping network variables 
        to their observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: float value indicating the VPI(potential | observed)
        """
        observed = self.meu(observed_evidence)
        pot_and_obs = dict(observed_evidence)
        potential_list = len(self.network.predict_proba(observed_evidence)[self.variable_names.index(potential_evidence)].items())
        meu_score = 0
        for val in range(potential_list):
            pot_and_obs[potential_evidence] = val
            potential = self.meu(pot_and_obs)
            meu_score += (potential[1]) * self.network.predict_proba(observed_evidence)[self.variable_names.index(potential_evidence)].items()[val][1]
            pot_and_obs.pop(potential_evidence)

        if meu_score - observed[1] >= 0:
            return meu_score - observed[1]
        else: return 0 


    def find_query(self, evidence, query) :
        return self.network.predict_proba(evidence)[query].items()
    

    def get_actions(self) :
        states = pd.DataFrame(self.data_file).nunique().to_dict()
        for k in states.keys():
            states[k] = range(states[k])
        if len(self.dec_vars) == 1 :
            return tuple(states[self.dec_vars[0]])
        a = {}
        for action in self.dec_vars:
            a[action] = states[action]
        return tuple(itertools.product(*(a[possibilites] for possibilites in a)))

    
    def find_EU(self, action, evidence, query_letter, index_of_letter) :
        updated_evid = dict(evidence)
        updated_evid.update(action)  
        queries = self.find_query(updated_evid, index_of_letter)
        util_scores = self.util_map.get(query_letter)
        avg_score = 0
        for i, query_val in queries : 
            avg_score += float(util_scores[i]) * float(query_val)
        return avg_score
    





        

class AdAgentTests(unittest.TestCase):

    def test_meu_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 0}, decision[0])
        self.assertAlmostEqual(2, decision[1], delta=0.01)
    
    def test_meu_lecture_example_with_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {"X": 0}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 1}, decision[0])
        self.assertAlmostEqual(2, decision[1], delta=0.01)
        
        evidence2 = {"X": 1}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"D": 0}, decision2[0])
        self.assertAlmostEqual(2.4, decision2[1], delta=0.01)
    
    def test_vpi_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        vpi = ad_engine.vpi("X", evidence)
        self.assertAlmostEqual(0.24, vpi, delta=0.1)
 
    def test_meu_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 0}, decision[0])
        self.assertAlmostEqual(746.72, decision[1], delta=0.01)
      
    def test_meu_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 1}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 1}, decision[0])
        self.assertAlmostEqual(720.73, decision[1], delta=0.01)
        
        evidence2 = {"T": 0, "G": 0}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"Ad1": 0, "Ad2": 0}, decision2[0])
        self.assertAlmostEqual(796.82, decision2[1], delta=0.01)
        
    def test_vpi_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(20.77, vpi, delta=0.1)
        
        vpi2 = ad_engine.vpi("F", evidence)
        self.assertAlmostEqual(0, vpi2, delta=0.1)
        
    def test_vpi_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 0}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(25.49, vpi, delta=0.1)
        
        evidence2 = {"G": 1}
        vpi2 = ad_engine.vpi("P", evidence2)
        self.assertAlmostEqual(0, vpi2, delta=0.1)
        
        evidence3 = {"H": 0, "T": 1, "P": 0}
        vpi3 = ad_engine.vpi("G", evidence3)
        self.assertAlmostEqual(66.76, vpi3, delta=0.1)
    
    def test_prob4(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        vpi3 = ad_engine.vpi("G", evidence)
        print(vpi3)
        return vpi3

    def test_prob4again(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"G" : 1}
        vpi3 = ad_engine.vpi("P", evidence)
        print(vpi3)
        return vpi3
    
if __name__ == '__main__':
    unittest.main()
