#!/usr/bin/python3
#################################################################################################
# This extraction code outputs the author and affiliation information in note tag of each entry.
#
# USAGE:
# 1. Run this code to create csv file from the input xml file in the argv[1].
# $  xml_sax_parse.py input_file.xml
#
# Output : "input_file_noTags.csv"
#
# Created by : Au Jit Seah
# File owners : Au Jit Seah
#################################################################################################
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

# For list update, "global" is unnecessary
source = []
target = []
affiliation_list = []

# Need to use "global" for update
sum_of_no_notes = 0


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
        filewriter.writerow(['Author', 'Institution'])
        
        iteration_range = graph_data.shape[0]
        for i in range(iteration_range):
            # Write from pdataFrame not pArray need to have iloc
            # print(i, graph_data.iloc[i, 0], graph_data.iloc[i, 1])
            filewriter.writerow([str(graph_data.iloc[i,0]), str(graph_data.iloc[i,1])])
            if i == iteration_range - 1:
                print('\nIndex that starts from 0 to', i)

        writeFile.close()

class DBLPHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.author = ''
        self.CurrentData = ''
        self.note = ''
        self.note_list = []
        self.affCnt = 0            # All publications with relevant xml tags
        self.total_no_notes = 0

        self.current_Tag = ''
        self.total_uninterested_venues = 0
        self.endElement_flag = False

    
        # For all tags : self.xml_tags = ['article', 'book', 'incollection', 'inproceedings', 'mastersthesis', 'phdthesis', 'proceedings', 'www']
        self.xml_tags = ['www']
        # '/inproceedings', '/proceedings', '/article', '/book', '/incollection', '/mastersthesis', '/phdthesis', '/www'
        # xml tags will not be parsed
        self.unwanted_xml_tags = ['article', 'book', 'incollection',
                                  'mastersthesis', 'phdthesis', 'inproceedings', 'proceedings']

        self.interested_details = False

   # (1) Call when an element starts
    def startElement(self, tag, attributes):
        # Sequence : characters -> startElement -> characters -> endElement
        # dblp XML order affects progressing sequence : www -> author -> title -> note -> url
        self.CurrentData = tag

        if tag in self.xml_tags:

            mdate = attributes['mdate']
            self.affCnt += 1

            print('\n------------------------------------------------------------------------------------------------------------------------------------')
            print('mdate :', mdate)
            print('Element No.  ' + str(self.affCnt) + ' (include without note)')
            print('Element Tag :', tag)
            
            # Important : Reset all variables after each round of interested xml tags
            self.interested_details = True
            # Clear note list for next www
            self.note_list = []
            # Reset author to prevent concatenation
            self.author = ''
            self.endElement_flag = False

            #if self.affCnt > 100:
                #exit()
        elif tag in self.unwanted_xml_tags:
            # Toggle back
            self.interested_details = False
            # Capture tag not in xml_tags but with note
            self.current_Tag = tag


    # (2) Call when a tag is read and store in relevant variables
    def characters(self, content):
        if self.interested_details == True:
            if self.CurrentData == "note":
                # Concatenation of the name with special characters "&amp" are crucial as it is not handled in "characters"
                # function via "Input Source" of the xml sax parser
                # Strip off an additional comma from the booktitle else there will be problem with CSV edgelist
                replaceStr = content.replace(',', '')

                self.note += replaceStr
            elif self.CurrentData == "author":
                # Replace "-" in the author name
                replaceStr = content.replace('-', ' ')

                #if self.interested_details == True:
                # Concatenation of the name with special characters "&#233" are crucial as it is not handled in "characters"
                # function via "Input Source" of the xml sax parser
                self.author += replaceStr


    # (3) Call when an elements ends
    def endElement(self, tag):
        # Reference to the global variable
        global sum_of_no_notes
        
        # CurrentData stores the tag which can be "dblpperson", "coauthors", "cite", "ee", "ref" for individual author's file
        # Capture information only for interested xml tags
        if self.interested_details == True:
            if self.CurrentData == "note":
                # Append note to the note list for each interested xml_tags
                self.note_list.append(self.note)
                affiliation_list.append(self.note)

                # Strip off an additional comma from note else there will be problem with CSV edgelist
                # replaceStr = content.replace(',', '')

                # Reset to uninterested details as venues are of no interest, no further processing required
                # if (replaceStr not in self.interested_venues):
                #     self.interested_details = False
                #     self.total_uninterested_venues += 1
                #     print("\nUninterested venue " + str(self.total_uninterested_venues) + " :", replaceStr)
                #
                # Update note for only interested xml_tags
                #if self.interested_details == True:
                    # booktitle_list.append(content)
                # else:
                #     print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                #     print("Booktitle \"" + content + "\" of tag \"" + self.current_Tag +
                #           "\" (different value for unwanted venues) is not in the required xml_tags or uninterested venues.")
                #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

                # Clear note to prevent note concatenation
                self.note = ""

                #print(self.note)
            elif self.CurrentData == "author":
                #print('{:s}'.format(self.author), end=', ')
                print("Author :", self.author)

            # "" captures end of complete tag (eg. "/www") before start of a new tag
            elif self.CurrentData == "":
                if self.endElement_flag == False:
                    print('No. of affiliations in note tag :', len(self.note_list))

                    for affiliation in self.note_list:
                        #print('{:s}'.format(self.note), end=', ')
                        print(affiliation)

                        # Create the source for the Author-Venue edgelist
                        source.append(self.author)
                        # Create the target for the Author-Venue edgelist
                        target.append(affiliation)

                    # This is the last element in "www" element tag
                    if len(self.note_list) == 0:
                        self.total_no_notes += 1
                        sum_of_no_notes = self.total_no_notes
                        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        print("Current total of www without notes :", sum_of_no_notes)
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                self.endElement_flag = True

        # Reset Current Data
        self.CurrentData = ''


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
    input("Press Enter to continue...\n")

    # Added for handling encoding; but not useful.  Need to build logic in "characters" function
    # with open(input_filename, "r") as readFile:
    #     input_source = xml.sax.xmlreader.InputSource()
    #     encoding = 'html'
    #     input_source.setCharacterStream(readFile) # setByteStream
    #     input_source.setEncoding(encoding)
    #
    #     print("Encoding", input_source.getEncoding())
    #
    #     parser.parse(input_source)

    # Take first argument as the input filename from sys.argv[1:]
    #parser.parse(io.open(input_filename))
    parser.parse(input_filename)

    # Hit any key to continue
    input("\nPress Enter to continue...")

    # Interested affiliation with note in www
    print('\nList of all authors with affiliation in note tag (non-unique) :\n', affiliation_list)
    print('\nTotal no. of affiliations in note tag :', len(affiliation_list))
    print('Total no. of affiliation in note tag with NO notes :', sum_of_no_notes)

    unique_values, counts = np.unique(affiliation_list, return_counts=True)
    print('The unique number of affiliations in note tag are :', len(unique_values))
    print('The unique values from affiliation_list are :', unique_values)
    # Gives different values from the degrees in the Author-Journal/Venue graph because authors give individual edges/links
    print('with the respective counts out of the total no. of affiliations :', counts)

    if len(unique_values) > 0:
        index_max = np.argmax(counts)
        print('Index of the highest count :', index_max)
        print('Value of the highest count :', unique_values[index_max])
    
    # Clear booktitle_list
    affiliation_list = []

    # Create graph data in csv file for graph building
    graph_data = pd.DataFrame(list(zip(source, target)), columns=['Author', 'Affiliation'])
    
    create_csv_file(output_filename_csv, graph_data)
    
    # Create graph from edgelist in the output csv file
    csv_filename_with_path = output_fullpath_csv_filename
    print('\nBuild graph from the output csv file :', csv_filename_with_path) # str(sys.argv[1])
    
    print('------------------------------------------------------------------------------------------------------------------------------------')
    secCmd = "ls -l %s %s" % (sys.argv[1], csv_filename_with_path)
    os.system(secCmd)
    os.system('echo \'EOF\'')
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print("End of Preprocessing\n")

# End of Program
