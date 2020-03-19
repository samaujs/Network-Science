#######################################################################
# This is the main function that invokes all the necessary procedures
# Created by : Au Jit Seah
# File owners : Au Jit Seah
#######################################################################
from science import *

#print("This is the first line of preprocessing.py!\n")

def retrieve_author_venue_info(author, venue):
    print("(3) Preprocessing ... please wait ...")
    print("Retrieving information of " + str(author) + " for " + str(venue) + " ...\n")
    result = "Degree Centrality"
    computeInfo(author, venue, result)
