#######################################################################
# This is the main function that invokes all the necessary procedures
# Created by : Au Jit Seah
# File owners : Anthony Koh Yao Wei
#######################################################################
from preprocessing import *

#print("This is the first line of GUI.\n")

def startGUI(xml_filename):
    print("(2) Starting GUI ...\n")
    
    # Obtain user input and call preprocessing.py or science.py
    author = "Sourav S. Bhowmick"
    venue = "VLDB"
    retrieve_author_venue_info(xml_filename, author, venue)

    # Display returned results from sources
