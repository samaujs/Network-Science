# Network Science
## Project 1 :
This project generates corresponding graphs for detailed network science analysis.  They are created from edgelist defined in the respective output CSV files via correspoding scripts. 

(A) Run ***"xml_sax_parser.py <xml filename>"*** will generate the author-venue graph information based on the provided venues of interest.<br>
User Inputs : <br>
(1) Provide the XML data file (eg. dblp.xml)<br>
(2) Provide the interested venues from a file/interface (eg. interested_venues.csv)<br>

Outputs : <br>
(1) CSV formatted file (eg. ./datasets/dblp.csv) to draw the author-venue graph.<br>

(B) Run ***"xml_sax_parser_institutes.py <xml filename>"*** will generate the author-institution graph information.<br>
User Inputs : <br>
(1) Provide the XML data file (eg. dblp.xml)<br>

Outputs : <br>
(1) CSV formatted file (eg. ./datasets/dblp.csv) to draw the author-institution graph.<br>

## Details
- Parse dblp XML data to obtain network science measures so as to justify observations.
- Extract the author, booktitle, venue and note information from the provided input xml file.
- Output respective edgelist CSV formatted files (based on desired scripts) to be used by Networkx function calls.
- Use the extracted information to build relevant graphs for visualization (Networkx. matplotlib).
- Perform Network Science analysis to substantiate findings from the extracted information. 
