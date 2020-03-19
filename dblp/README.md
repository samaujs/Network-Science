# Network Science
## Project 1 :
Parse dblp XML data to obtain network science measures so as to justify observations.
This project attempts to build corresponding graphs from edgelist defined in the respective output CSV files for detailed analysis. 

User Inputs : <br />
(1) Provide the XML data file (eg. dblp.xml)<br />
(2) Provide the interested venues (eg. interested_venues.csv)<br />

Outputs : <br />
(1) CSV formatted file (eg. dblp_noTags.csv) to draw the author-venue graph.

## Details
- Extract the authors and booktitle information for conferences from the provided input xml file.
- Output an edgelist CSV formatted file to be used by Networkx function calls.
- Use the extracted information to build graphs for visualization (Networkx. matplotlib).
- Perform Network Science analysis to substantiate findings from the extracted information. 
