# -------------------------------------------------------------------------
# AUTHOR: Cristopher Hernandez
# FILENAME: similarity.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open("cleaned_documents.csv", "r") as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append(row)
         print(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here
doc_term_matrix = []
tokenized_documents = []
terms_map = {}

term_index = 0
document_id: str
document_content: str
for document_id, document_content in documents:
    tokenized_content = document_content.split(' ')
    tokenized_documents.append(
        (document_id, tokenized_content)
    )
    for term in tokenized_content:
        if term not in terms_map:
            terms_map[term] = term_index
            term_index += 1

for document_id, tokenized_content in tokenized_documents:
    document_data = [0] * len(terms_map)
    for term in tokenized_content:
        if document_data[terms_map[term]] < 1:
            document_data[terms_map[term]] = 1
    doc_term_matrix.append(
        (document_id, document_data)
    )

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
highest_similarity = -999
best_pair = None

for document_id_a, document_data_a in doc_term_matrix:
    print(f"Comparing document {document_id_a} to rest of collection...")
    for document_id_b, document_data_b in doc_term_matrix:
        if document_id_a == document_id_b:
            # Skip over the same document
            continue
        similarity = cosine_similarity([document_data_a], [document_data_b])
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_pair = (document_id_a, document_id_b)

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
if best_pair is not None:
    print(
        f"The most similar documents are document {best_pair[0]} "
        f"and document {best_pair[1]} with cosine similarity = {highest_similarity}"
    )
else:
    print("Something went wrong: no best pair found.")
