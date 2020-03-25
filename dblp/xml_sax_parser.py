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

# For list update, "global" is unnecessary
source = []
target = []
booktitle_list = []

# Need to use "global" for update
sum_of_no_authors = 0


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
                print('\nIndex that starts from 0 to', i)

        writeFile.close()

class DBLPHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.author = ''
        self.title = ''
        self.year = ''
        self.booktitle = ''
        self.CurrentData = ''
    
        self.author_list = []
        self.author = 'Sam'

        self.pubCnt = 0            # All publications with relevant xml tags
        self.lastBookTitle = ''
        self.total_no_authors = 0
        
    #unwanted_tagsCnt = 0
    #notUsedBookTitle = ''
    #notUsedBooktitleList = []
    
        # For all tags : self.xml_tags = ['article', 'book', 'incollection', 'inproceedings', 'mastersthesis', 'phdthesis', 'proceedings', 'www']
        self.xml_tags = ['inproceedings', 'proceedings']
        self.unwanted_xml_tags = ['/inproceedings', '/proceedings', 'article', '/article', 'book', '/book', 'incollection', '/incollection',
                                  'mastersthesis', '/mastersthesis','phdthesis', '/phdthesis', 'www', '/www']

        self.interested_details = False

   # Call when an element starts
    def startElement(self, tag, attributes):
        # Sequence : characters -> startElement -> characters -> endElement
        self.CurrentData = tag

        if tag in self.xml_tags:

            mdate = attributes['mdate']
            self.pubCnt += 1

            print('------------------------------------------------------------------------------------------------------------------------------------')
            print('Mdate :', mdate)
            print('Element No. :', self.pubCnt)
            print('Element Tag :', tag)
            
            # Important : Reset all variables after each set
            self.lastBookTitle = ''

            self.interested_details = True

            #if self.pubCnt > 100:
                #exit()
        elif tag in self.unwanted_xml_tags:
            # Toggle back
            self.interested_details = False


    # Call when an elements ends
    def endElement(self, tag):
        # Reference to the global variable
        global sum_of_no_authors
        
        # CurrentData stores the tag which can be "dblpperson", "coauthors", "cite", "ee", "ref" for individual author's file
        # Capture information only for interested xml tags
        if self.interested_details == True :
            if self.CurrentData == "year":
                #print("Year :", self.year)
                print('{:s}'.format(self.year), end=', ')
            elif self.CurrentData == "booktitle":
                print('\nNo. of authors :', len(self.author_list))
                
                for author in self.author_list:
                    # Create the source for the Author-Venue edgelist
                    source.append(author)
                    # Create the target for the Author-Venue edgelist
                    target.append(self.booktitle)
            
                if len(self.author_list) == 0:
                    self.total_no_authors += 1
                    sum_of_no_authors = self.total_no_authors
                
                # Clear author list for next booktitle
                self.author_list = []
                print(self.booktitle)
                #print('{:s}'.format(self.booktitle), end=', ')
            elif self.CurrentData == "author":
                #print("Author :", self.author)
                print('{:s}'.format(self.author), end=', ')
            elif self.CurrentData == "title":
                print('\n{:s}'.format(self.title), end=', ')

        # Reset Current Data
        self.CurrentData = ''

    # Call when a tag is read and store in relevant variables
    def characters(self, content):
        if self.CurrentData == "year":
            #print("Content :", content)
            self.year = content
        elif self.CurrentData == "booktitle":
            # Strip off an additional comma from the booktitle else there will be problem with CSV edgelist
            replaceStr = content.replace(',','')

            if self.lastBookTitle == '':
                booktitle_list.append(replaceStr)
                self.lastBookTitle = replaceStr
            
            self.booktitle = replaceStr
        elif self.CurrentData == "author":
            oldStr = content
            replaceStr = oldStr.replace('-',' ')
            # Append author list only for every new booktitle and when xml_tags are matched
            if self.lastBookTitle == '' and self.interested_details == True :
                self.author = replaceStr
                # Append author for new booktitle
                self.author_list.append(self.author)
            
            self.author = replaceStr
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
    input("Press Enter to continue...\n")
    
    # Take first argument as the input filename from sys.argv[1:]
    # parser.parse(io.open("./BhowmickSouravS_noTags.xml"))
    parser.parse(input_filename)

    # Hit any key to continue
    input("Press Enter to continue...\n")
    
    # Interested conferences/venues with booktitle
    print('\nList of all interested conferences/venues with booktitles (non-unique) :\n', booktitle_list)
    print('\nTotal no. of interested conferences/venues with booktitles :', len(booktitle_list))
    print('Total no. of booktitles with NO authors :', sum_of_no_authors)

    unique_values, counts = np.unique(booktitle_list, return_counts=True)
    print('The unique number of booktitles/venues are :', len(unique_values))
    print('The unique values from booktitle_list are :', unique_values)
    # Gives different values from the degrees in the Author-Journal/Venue graph because authors give individual edges/links
    print('with the respective counts out of the total no. of publications :', counts)

    index_max = np.argmax(counts)
    print('Index of the highest count :', index_max)
    print('Value of the highest count :', unique_values[index_max])
    
    # Clear booktitle_list
    booktitle_list = []

    # Create graph data in csv file for graph building
    graph_data = pd.DataFrame(list(zip(source, target)), columns=['Author', 'Venue'])
    
    create_csv_file(output_filename_csv, graph_data)
    
    # Create graph from edgelist in the output csv file
    csv_filename_with_path = output_fullpath_csv_filename
    print('\nBuild graph from the output csv file :', csv_filename_with_path) # str(sys.argv[1])
    
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
    
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print("End of Preprocessing\n")

# End of Program
