#!/usr/bin/python3
#######################################################################################
# This extraction code outputs the author and booktitle information of each entry.
#
# USAGE:
# 1. Run this code to create csv file from the input xml file in the argv[1].
# $  xml_sax_parse.py input_file.xml
#
# Output : "input_file_noTags.csv"
#
# Created by : Au Jit Seah
# File owners : Au Jit Seah
########################################################################################
import xml.sax
import sys
import os
import numpy as np
import pandas as pd
import csv

# Global variables
# Where to save the figures
PROJECT_ROOT_DIR = "."
FOLDER = "datasets"
output_filename_notags_csv = ""
output_fullpath_csv_filename = ""

source = []
target = []

def load_data(filename, data_path=output_fullpath_csv_filename):
    csv_path = os.path.join(data_path, filename)
    return pd.read_csv(csv_path)

# Create csv file
def create_csv_file(filename, graph_data):
    data_path = os.path.join(PROJECT_ROOT_DIR, FOLDER)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    # Create and overwrite existing file
    with open(FOLDER + '/'+ filename, 'w') as writeFile:
        filewriter = csv.writer(writeFile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Author', 'Venue'])
        
        iteration_range = graph_data.shape[0]
        for i in range(iteration_range):
            # Write from pdataFrame not pArray need to have iloc
            # print(i, graph_data.iloc[i, 0], graph_data.iloc[i, 1])
            filewriter.writerow([str(graph_data.iloc[i,0]), str(graph_data.iloc[i,1])])
            if i == iteration_range - 1:
                print("\nIndex that starts from 0 to", i)

        writeFile.close()

class DBLPHandler( xml.sax.ContentHandler ):

    def __init__(self):
        self.CurrentData = ""
        self.year = ""
        self.booktitle = ""
        self.author = ""
        self.title = ""
        self.count = 0
    
        #self.xml_tags = ['article', 'book', 'incollection', 'inproceedings', 'mastersthesis', 'phdthesis', 'proceedings', 'www']
        self.xml_tags = ['inproceedings', 'proceedings']
        self.xml_end_tags = ['/inproceedings', '/proceedings', 'article', '/article', 'book', '/book',
                             'incollection', '/incollection', 'mastersthesis', '/mastersthesis','phdthesis', '/phdthesis', 'www', '/www']

        self.interested_details = False

   # Call when an element starts
    def startElement(self, tag, attributes):
        # Sequence : characters -> startElement -> characters -> endElement
        self.CurrentData = tag

        if tag in self.xml_tags:
            mdate = attributes["mdate"]
            self.count += 1

            print("------------------------------------------------------------------------------------------------------------------------------------")
            print("Mdate : ", mdate)
            print("Element No. :", self.count)
            print("Element Tag :", tag)

            self.interested_details = True

            #if self.count > 100:
                #exit()
        elif tag in self.xml_end_tags:
            # Toggle back
            self.interested_details = False


    # Call when an elements ends
    def endElement(self, tag):
        # CurrentData stores the tag which can be "dblpperson", "coauthors", "cite", "ee", "ref" for individual author's file
        if self.interested_details == True :
            if self.CurrentData == "year":
                #print("Year :", self.year)
                print('{:s}'.format(self.year), end=', ')
            elif self.CurrentData == "booktitle":
                print(self.booktitle)
                #print('{:s}'.format(self.booktitle), end=', ')
            elif self.CurrentData == "author":
                #print("Author :", self.author)
                print('{:s}'.format(self.author), end=', ')
            elif self.CurrentData == "title":
                #print("Title :", self.title)
                print('\n{:s}'.format(self.title), end=', ')
#            else:
#                print("**************************** No Value ****************************")
#        elif tag in self.xml_end_tags:
#                # Toggle back
#                self.interested_details = False

        # Reset Current Data
        self.CurrentData = ""

    # Call when a tag is read and store in relevant variables
    def characters(self, content):
        if self.CurrentData == "year":
            #print("Content :", content)
            self.year = content
        elif self.CurrentData == "booktitle":
            self.booktitle = content
        elif self.CurrentData == "author":
            self.author = content
        elif self.CurrentData == "title":
            self.title = content
  
if ( __name__ == "__main__"):
   
    input_filename = sys.argv[1]
    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # override the default ContextHandler
    Handler = DBLPHandler()
    parser.setContentHandler( Handler )
   
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print("START OF PROGRAM")
    
    input_filename_noExt, file_extension = os.path.splitext(sys.argv[1])
    output_filename_csv = input_filename_noExt + ".csv"
    output_fullpath_csv_filename = os.path.join(FOLDER, output_filename_csv)

    # Execution is asynchronous
    # Hit any key to continue
    print("Input filename with no extension :", input_filename_noExt)
    input("Press Enter to continue...")
    
    # Take first argument as the input filename from sys.argv[1:]
    # parser.parse(io.open("./BhowmickSouravS_noTags.xml"))
    parser.parse(input_filename)

    # Create graph data in csv file for graph building
    graph_data = pd.DataFrame(list(zip(source, target)), columns=['Author', 'Venue'])
    
    create_csv_file(output_filename_csv, graph_data)
    
    # Create graph from edgelist in the output csv file
    csv_filename_with_path = output_fullpath_csv_filename
    print("\nBuild graph from the output csv file :", csv_filename_with_path) # str(sys.argv[1])
    
    # Hit any key to continue
    input("Press Enter to continue...")
    
    os.system('ls -tlag')
    print("\n\nExecute more unix command above!")
    
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print("END OF PROGRAM\n\n")
    
    os.system('echo \'System command executes faster than python script!!!\'')
    secCmd = "ls -l %s %s" % (sys.argv[1], csv_filename_with_path)
    os.system(secCmd)
    os.system('echo \'EOF\'')
    
    print("------------------------------------------------------------------------------------------------------------------------------------")
    print("End of Preprocessing\n")

# End of Program
