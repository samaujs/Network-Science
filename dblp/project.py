#!/usr/bin/python3
#######################################################################
# This is the main function that invokes all the necessary procedures
# Created by : Au Jit Seah
# File owners : Au Jit Seah
#######################################################################
import sys
from interface import startGUI

print("------------------------------------------")
print("|  Welcome to Network Science Project 1  |")
print("------------------------------------------")

if len(sys.argv) < 2:
    print("USAGE : ./project.py input_file.xml")
    print("Please provide the xml filename for preprocessing.\n")
    exit()

xml_filename = sys.argv[1]
print("(1) This is where we start with user input file \"" + xml_filename + "\" .....\n")

startGUI(xml_filename)
