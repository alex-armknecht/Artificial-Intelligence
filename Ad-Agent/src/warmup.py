from pomegranate import *
import pandas as pd
'''
warmup.py

Skeleton for answering warmup questions related to the
AdAgent assignment. By the end of this section, you should
be familiar with:
- Importing, selecting, and manipulating data using Pandas
- Creating and Querying a Bayesian Network
- Using Samples from a Bayesian Network for Approximate Inference

@author: <Alex Armknecht, Makena Robison, Emilie Coon>
'''

if __name__ == '__main__':
    """
    PROBLEM 2.1
    Using the Pomegranate Interface, determine the answers to the
    queries specified in the instructions.
    
    ANSWER GOES BELOW:
    @ an index.items will give you the query for it 

    𝑃(𝑆|𝐴𝑑1=0,𝐴𝑑2=0) = 
    ((0, 0.5210997268105008), (1, 0.18271609537196118), (2, 0.29618417781753803))

    P(𝑆|𝐺=1,𝐴𝑑1=0,𝐴𝑑2=1) = 
    ((0, 0.6042203715683895), (1, 0.07783401272929892), (2, 0.3179456157023116))

    P(𝑆|𝑇=1,𝐻=1,𝐴𝑑1=1,𝐴𝑑2=0) =
    ((0, 0.29728928811552147), (1, 0.360204174806146), (2, 0.3425065370783326))

    """
    X = pd.read_csv('../dat/adbot-data.csv') #load in that data bestie
    variable_names = list(X.columns) # put in columns so easier to read #good
    print(variable_names)
    print(X.head()) # numbers the columns
    network = BayesianNetwork.from_samples(X, algorithm = 'exact', state_names = variable_names) #create BN
    query =  network.predict_proba({"Ad1" : 0,"Ad2" : 0 })[5].items() #filler example [3].items()
    query2 = network.predict_proba({"G" : 1, "Ad1" : 0,"Ad2" : 1 })[5].items()
    query3 = network.predict_proba({"T" : 1, "H" : 1, "Ad1" : 1,"Ad2" : 0 })[5].items()
    print(query)
    print(query2)
    print(query3)


    
