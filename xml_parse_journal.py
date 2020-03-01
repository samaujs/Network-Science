#
# USAGE:
#
# This code outputs the author and journal information of each entry.
#
# 1. Preprocess dblp.xml and make dblp_noTags.xml, which is parsed by this code. Replaces all the unwanted tags to avoid unrecognised titles.
# $ cat dblp.xml | sed 's/<i>//g' | sed 's/<\/i>//g' | sed 's/<sup>//g' | sed 's/<\/sup>//g' | sed 's/<sub>//g' | sed 's/<\sub>//g' | sed 's/<tt>//g' | sed 's/<\/tt>//g' > input_file_notags.xml
#
# 2. Create tags that begins with "</" in store them in "xml_tags.txt", which will be read in this code.
# $ cat dblp.xml | awk '{if(substr($1,1,2)=="</"){split($1,a,">");print substr(a[1],3,length(a[1]))}}' | uniq | sort | uniq > xml_tags.txt
#
# 3. Run this code to create csv file from the input xml file in the argv[1].
# $ python xml_parse_journal.py input_file.xml
#

from lxml import etree
import os
import sys
from io import TextIOWrapper
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
import csv

sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Global variables
DATA_PATH = os.path.join("datasets", "dblp")
source = []
target = []

tokenizer = RegexpTokenizer(r'\w+')

with open('xml_tags.txt') as f:
    xml_tags = f.read().splitlines()


def load_data(filename, data_path=DATA_PATH):
    csv_path = os.path.join(data_path, filename)
    return pd.read_csv(csv_path)

# Create csv file 
def create_csv_file(filename, graph_data):
    
    # Create and overwrite existing file
    with open('datasets/'+ filename, 'w') as writeFile:
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

# Extracting the element information
def extract_elem_info(context):
    author_list = []
    title = ''
    year = ''
    journal = ''
    journal_list = []
    counter = 0

    #read chunk line by line
    #we focus author and title
    for event, elem in context :
        if elem.tag == 'title' :
            if elem.text :
                title = elem.text
        if elem.tag == 'year' :
            if elem.text :
                year = elem.text

        if elem.tag == 'author' :
            if elem.text :
                oldStr = elem.text
                replaceStr = oldStr.replace('-',' ')
                author_list.append(replaceStr)

        # If journal does not exist, then there will not be any element text
        if elem.tag == 'journal' :
            if elem.text :
                journal = elem.text
                journal_list.append(elem.text)

        if elem.tag in xml_tags :
            # and ('Inf. Sci.' in journal or 'Image Vision Comput.' in journal or 'VLDB' in journal) :
            if title and year :
                counter += 1
                print("Element No. :", counter)

                year = int(year)
                print('{:d}'.format(year), end='')

                # For all publications from author
                if journal :
                    print(', {:s}'.format(journal), end=', ')
                else :
                    print(', NoJ', end=', ')
                    # If journal does not exist for the elem.tag just included no journal present in the journal_list
                    journal_list.append('NoJ')

                # Authors
                for author in author_list:
                    print('{:s}'.format(author), end=', ')

                    # Create the Author-Journal/Venue lists
                    source.append(author)
                    if journal :
                        target.append(journal)
                    else :
                        target.append('NoJ')

                print('')

                print("No. of authors :", len(author_list))

                for word in tokenizer.tokenize(title):
                    print('{:s}'.format(word), end=' ')
                #print("\nSource :", source)
                #print("\nTarget :", target)

                print(flush=True)
                
                print('Element Tag :', elem.tag)
                #print('Journal list:', journal_list)
                print('------------------------------------------------------------------------------------------------------------------------------------')

                title = ''
                year = ''
                journal = ''

            # Clear all the authors
            author_list = []

        elem.clear()

        # while elem.getprevious() is not None:
            # del elem.getparent()[0]
            # print(elem.getprevious())
    del context
    print('\nList of all publications :', journal_list)
    print('\nTotal no. of publications :', len(journal_list))

    counter = 0
    unique_values, counts = np.unique(journal_list, return_counts=True)
    print("The unique number of journals/venues are :", len(unique_values))
    print("The unique values from journal_list are :", unique_values)
    # Gives different values from the degrees in the Author-Journal/Venue graph because authors give individual edges/links
    print("with the respective counts out of the total no. of publications :", counts)

    index_max = np.argmax(counts)
    print("Index of the highest count :", index_max)
    print("Value of the highest count :", unique_values[index_max])

    journal_list = []

    # Hit any key to continue
    input("Press Enter to continue...")

    
# Start of the main program
if __name__ == "__main__":
    # Usage : python xml_parse_journal.py <XML filename>
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print("START OF PROGRAM")

    input_filename_noExt, file_extension = os.path.splitext(sys.argv[1])
    output_filename = input_filename_noExt + "_noTags.xml"

    cmd = "cat %s | sed \'s/<i>//g\' | sed \'s/<\/i>//g\' | sed \'s/<sup>//g\' | sed \'s/<\/sup>//g\' | sed \'s/<sub>//g\' | sed \'s/<\sub>//g\' | sed \'s/<tt>//g\' | sed \'s/<\/tt>//g\' > %s" % (sys.argv[1], output_filename)
    os.system(cmd)

    # Execution is asynchronous
    # Hit any key to continue
    print("Input filename with no extension :", input_filename_noExt)
    print("NoTags output filename with XML extension :", output_filename)
    input("Press Enter to continue...")

    # Take first argument as the input filename from sys.argv[1:]
    context = etree.iterparse(output_filename, load_dtd=True, html=True) # sys.argv[1]
    extract_elem_info(context)

    # Create graph data in csv file for graph building
    graph_data = pd.DataFrame(list(zip(source, target)), columns=['Author', 'Venue'])

    create_csv_file(input_filename_noExt + "_noTags.csv", graph_data)

    # Create graph from edgelist in the output csv file
    csv_filename_with_path = "./datasets/" + input_filename_noExt + "_noTags.csv"
    print("\nBuild graph from the output csv file :", csv_filename_with_path) # str(sys.argv[1])

    # Hit any key to continue
    input("Press Enter to continue...")

    os.system('ls -tlag')
    print("\n\nExecute more unix command above!")

    print('------------------------------------------------------------------------------------------------------------------------------------')
    print("END OF PROGRAM\n\n")

    os.system('echo \'System command executes faster than python script!!!\'')
    secCmd = "ls -l %s %s %s" % (sys.argv[1], output_filename, csv_filename_with_path)
    os.system(secCmd)
    os.system('echo \'EOF\'')

# End of Program
