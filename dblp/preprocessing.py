#######################################################################
# This is the main function that invokes all the necessary procedures
# Created by : Au Jit Seah
# File owners : Au Jit Seah
#######################################################################
import os

from science import *

#print("This is the first line of preprocessing.py!\n")

def retrieve_author_venue_info(xml_filename, author, venue):
    print("(3) Preprocessing ... please wait ...")
    print("Retrieving information of " + str(author) + " for " + str(venue) + " in \"" + xml_filename + "\" ...\n")
    result = "Degree Centrality"
    computeInfo(author, venue, result)

    # filename is passed from main program
    cmd = "./xml_sax_parser.py " + xml_filename
    os.system(cmd)
