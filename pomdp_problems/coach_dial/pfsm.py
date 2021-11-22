import random
import math
import sys
import copy
import argparse
import os
import logging

MAX_INSTRUCTION_STEP = 2

states=[(i=1,world=0000,actv=doing-nothing), 
        ((i=1,world=0001,actv=doing-nothing), 
        (i=1,world=0000,actv=doing-nothing, act =putting-saucepan-on-stove"))
        (i=1,world=0001,actv=putting-saucepan-on-stove),
        ]


class pfsm:
    def __init__(self, specification):
        self.specification = specification
        self.states=
    
    def adjacency_matrix(self, specification):
        self.matrix = {states : wait, bye, give_next
        s0: [s0,s1,s3], ,[s0,s4],
        s1:[]
        s2:  }

    def output_csv(self, matrix):
        read_matrix
        write.csv()

        




if __name__ == "__main__":

    ###
    specification={
        #instruction_step :{in}
        1:{"w":{'0000'}, "actv":{"putting-saucepan-on-stove"}}, 
        2:{"w":{'0001'}, "actv": {"adding-oil-in-the-pan"}},
    }