import numpy as np
import re as re
import pandas as pd
import skbio as skbio
import itertools as itertools
from Bio import SeqIO

from ..get_orfs import find_next_stop

def test_next_stop_no_longest():
    "check this finds the NEXT stop codon"
    assert find_next_stop("AAAMBBB*CCC*", 4) == 8

def test_next_stop_final():
    "check that this returns the length of the given string when no stop codon is found"
    assert find_next_stop("AAAMBBBCCC", 4) == 14
