# Quotation annotations

Annotations are contained in .tsv files, which have two components (quote annotations and speaker attribution annotations); each type has its own structure:

|Label|Quote ID|Start sentence ID|Start token ID (within sentence)|End sentence ID|End token ID (within sentence)|Quotation|
|---|---|---|---|---|---|---|
QUOTE|Q342|54|0|54|13|“ Of course , we ’ll take over your furniture , mother , ”|


|Label|Quote ID|Speaker ID|
|---|---|---|
|ATTRIB|Q342|Winnie_Verloc-3|

Sentence IDs correspond to the row of the sentence in the corresponding .txt file; token IDs correspond to index of the token within that sentence.