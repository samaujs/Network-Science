#
# USAGE:
#
# This code outputs the author and booktitle information of each entry.
#
# 1. Preprocess dblp.xml and make dblp_noTags.xml, which is parsed by this code. Replaces all the unwanted tags to avoid unrecognised titles.
# $ cat dblp.xml | sed 's/<i>//g' | sed 's/<\/i>//g' | sed 's/<sup>//g' | sed 's/<\/sup>//g' | sed 's/<sub>//g' | sed 's/<\sub>//g' | sed 's/<tt>//g' | sed 's/<\/tt>//g' > input_file_notags.xml
#
# 2. Create tags that begins with "</" in store them in "xml_tags.txt", which will be read in this code.
# $ cat dblp.xml | awk '{if(substr($1,1,2)=="</"){split($1,a,">");print substr(a[1],3,length(a[1]))}}' | uniq | sort | uniq > xml_tags.txt
#
# 3. Run this code to create csv file from the input xml file in the argv[1].
# $ python xml_parse_conferences.py input_file.xml
#
# Output : "input_file_noTags.csv" and "input_file_noTags.xml"

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
# Where to save the figures
PROJECT_ROOT_DIR = "."
FOLDER = "datasets"
output_filename_notags_csv = ""
output_fullpath_csv_filename = ""

source = []
target = []

tokenizer = RegexpTokenizer(r'\w+')

with open('xml_tags.txt') as f:
    xml_tags = f.read().splitlines()


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

# Return intersection list
def intersection(list1, list2):
    return list(set(list1) & set(list2))

# Extracting the element information
def extract_elem_info(context):
    author_list = []
    title = ''
    year = ''
    booktitle = ''
    booktitle_list = []
    pubCnt = 0            # All publications
    confCnt = 0           # Interested conferences/venues with booktitle same as len(booktitle_list)

    unwanted_tagsCnt = 0
    notUsedBookTitle = 'PastThroughTheFirstTime'
    notUsedBooktitleList = []
    lastBookTitle = ''

    unwanted_tags = ['article', 'book', 'incollection', 'mastersthesis', 'phdthesis', 'www']

    #read blocks line by line, look for author and booktitle
    for event, elem in context :

        # Need to capture information before interested xml_tag appears
        # "inproceedings" appear in elem.tag only when </inproceedings> appear
        # (1) author pid
        if elem.tag == 'author':
            if elem.text:
                oldStr = elem.text
                replaceStr = oldStr.replace('-', ' ')

                # Append author list only for every new title
                if booktitle == '':
                    author_list.append(replaceStr)

        # (2) title
        if elem.tag == 'title':
            if elem.text:
                title = elem.text
                pubCnt += 1

        # (3) year
        if elem.tag == 'year':
            if elem.text:
                year = elem.text

        # (4) booktitle
        # If booktitle does not exist, then there will not be any element text
        if (elem.tag == 'booktitle') and (lastBookTitle == ''):
            if elem.text:
                booktitle = elem.text
                lastBookTitle = booktitle

        if elem.tag in xml_tags :
            # Now to store required information for interested xml_tags
            if booktitle and year:
                confCnt += 1
                print("Element No. :", confCnt)

                year = int(year)
                print('{:d}'.format(year), end='')

                # For all conferences/venues based on booktitle from author
                print(', {:s}'.format(booktitle), end='\n')
                booktitle_list.append(booktitle)

                # Append author list only for every new title and when xml_tags are matched
                for author in author_list:
                    print('{:s}'.format(author), end=', ')

                    # Create the Author-Journal/Venue lists
                    source.append(author)
                    target.append(booktitle)

                print('')
                print("No. of authors :", len(author_list))

                # Show the title with the booktitle/venue/conferences
                for word in tokenizer.tokenize(title):
                    print('{:s}'.format(word), end=' ')
                #print("\nSource :", source)
                #print("\nTarget :", target)

                print('\nElement Tag :', elem.tag)
                #print('booktitle list:', booktitle_list)
                print('------------------------------------------------------------------------------------------------------------------------------------')
                print(flush=True)

                # Reset all variables
                title = ''
                year = ''
                booktitle = ''
                lastBookTitle = ''
                author_list = []

                # Only interested in tags with booktitles
                # elif not booktitle :
                #     print(', NoBt', end=', ')
                #     # If booktitle does not exist for the elem.tag just included no booktitle present in the booktitle_list
                #     booktitle_list.append('NoBt')
                #     target.append('NoBt')
            else:
                print('------------------------------------------------------------------------------------------------------------------------------------')
                print("Missing year : " + year + " or missing title : " + booktitle)

                # Need to clear author list if required information is incomplete
                #author_list = []
        else:
            if (elem.tag in unwanted_tags) :

                if notUsedBookTitle != booktitle and booktitle:
                    unwanted_tagsCnt += 1
                    notUsedBookTitle = booktitle
                    notUsedBooktitleList.append(booktitle)

                    #lastBookTitle = ''

                    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    print("Tag not in xml_tags : " + elem.tag + ", with title : " + title + ", on : " + year)
                    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

                # Need to clear list otherwise the author list will be concatenated
                author_list = []
                # Reset booktitle so that new author can be appended to list for new interested xml_tags
                booktitle = ''
                lastBookTitle = ''

        elem.clear()

        # while elem.getprevious() is not None:
            # del elem.getparent()[0]
            # print(elem.getprevious())
    del context

    # Hit any key to continue
    input("Press Enter to continue...")

    print('\nList of all interested conferences/venues with booktitles (non-unique) :\n', booktitle_list)
    print('\nTotal no. of interested conferences/venues with booktitles :', len(booktitle_list))
    print('\nList of book titles not in the required xml_tags :\n', notUsedBooktitleList)
    print('\nTotal no. of interested conferences/venues with unwanted tags :', unwanted_tagsCnt)

    # dblp refined by "Books and Theses, Conferences and Workshop Papers,
    # Parts in Books or Collections, Editorship, Reference Works"
    # 2 elements, b1 and b2 in "Books and Theses" are without booktitle making total counts of 176 instead of 178
    print("\nNo. of Booktitles (starts from 0) :", confCnt)

    pubCnt = 0
    confCnt = 0
    unique_values, counts = np.unique(booktitle_list, return_counts=True)
    print("The unique number of booktitles/venues are :", len(unique_values))
    print("The unique values from booktitle_list are :", unique_values)
    # Gives different values from the degrees in the Author-Journal/Venue graph because authors give individual edges/links
    print("with the respective counts out of the total no. of publications :", counts)

    index_max = np.argmax(counts)
    print("Index of the highest count :", index_max)
    print("Value of the highest count :", unique_values[index_max])

    booktitle_list = []

    
# Start of the main program
if __name__ == "__main__":
    # Usage : python xml_parse_conferences.py <XML filename>
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print("START OF PROGRAM")

    input_filename_noExt, file_extension = os.path.splitext(sys.argv[1])
    output_filename_notags_csv = input_filename_noExt + "_noTags.csv"
    output_fullpath_csv_filename = os.path.join(FOLDER, output_filename_notags_csv)
    output_xml_filename = input_filename_noExt + "_noTags.xml"

    cmd = "cat %s | sed \'s/<i>//g\' | sed \'s/<\/i>//g\' | sed \'s/<sup>//g\' | sed \'s/<\/sup>//g\' | sed \'s/<sub>//g\' | sed \'s/<\sub>//g\' | sed \'s/<tt>//g\' | sed \'s/<\/tt>//g\' > %s" % (sys.argv[1], output_xml_filename)
    #os.system(cmd)

    # Execution is asynchronous
    # Hit any key to continue
    print("Input filename with no extension :", input_filename_noExt)
    print("NoTags output filename with XML extension :", output_xml_filename)
    input("Press Enter to continue...")

    # Take first argument as the input filename from sys.argv[1:]
    context = etree.iterparse(output_xml_filename, load_dtd=True, html=True) # sys.argv[1]
    extract_elem_info(context)

    # Create graph data in csv file for graph building
    graph_data = pd.DataFrame(list(zip(source, target)), columns=['Author', 'Venue'])

    create_csv_file(output_filename_notags_csv, graph_data)

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
    secCmd = "ls -l %s %s %s" % (sys.argv[1], output_xml_filename, csv_filename_with_path)
    os.system(secCmd)
    os.system('echo \'EOF\'')

# End of Program
