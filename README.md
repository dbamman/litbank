# LitBank


LitBank is an annotated dataset of 100 works of English-language fiction to support tasks in natural language processing and the computational humanities, described in more detail in the following publications:

* David Bamman, Sejal Popat and Sheng Shen (2019), "[An Annotated Dataset of Literary Entities](http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/naacl2019_literary_entities.pdf)," NAACL 2019.

* Matthew Sims, Jong Ho Park and David Bamman (2019), "[Literary Event Detection](http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/acl2019_literary_events.pdf)," ACL 2019.

* David Bamman, Olivia Lewke and Anya Mansoor (2020), "[An Annotated Dataset of Coreference in English Literature](https://arxiv.org/abs/1912.01140)", LREC.

LitBank currently contains annotations for entities, events, entity coreference, and quotation attribution in a sample of ~2,000 words from each of those texts, totaling 210,532 tokens.


<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Dataset" property="dct:title" rel="dct:type">LitBank</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Entity annotations

The entity annotation layer of LitBank covers six of the ACE 2005 categories in text:

* People (PER): *Tom Sawyer*, *her daughter*
* Facilities (FAC): *the house*, *the kitchen*
* Geo-political entities (GPE): *London*, *the village*
* Locations (LOC): *the forest*, *the river*
* Vehicles (VEH): *the ship*, *the car*
* Organizations (ORG): *the army*, *the Church*

The targets of annotation here include both named entities (e.g., Tom Sawyer) and common entities (the boy).  These entities can be nested, as in the following:

<img src="img/nested_structure.png" alt="drawing" width="300"/>


For more, see: David Bamman, Sejal Popat and Sheng Shen, "[An Annotated Dataset of Literary Entities](http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/naacl2019_literary_entities.pdf)," NAACL 2019.


## Event annotations

The event layer in LitBank identifies events with asserted *realis* (depicted as actually taking place, with specific participants at a specific time) -- as opposed to events with other epistemic modalities (hypotheticals, future events, extradiegetic summaries by the narrator).

|Text|Events|Source|
|---|---|---|
|My father’s eyes had **closed** upon the light of this world six months, when mine **opened** on it.|{closed, opened}|Dickens, David Copperfield|
|Call me Ishmael.|{}|Melville, Moby Dick|
|His sister was a tall, strong girl, and she **walked** rapidly and resolutely, as if she knew exactly where she was going and what she was going to do next.|{walked}|Cather, O Pioneers|

For more, see: Matt Sims, Jong Ho Park and David Bamman, "[Literary Event Detection](http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/acl2019_literary_events.pdf)," ACL 2019.

## Coreference annotations

The coreference layer in LitBank covers the six ACE entity categories outlined above (people, facilities, locations, geo-political entities, organizations and vehicles); while the entity tagging above only covers proper noun phrases (*Tom Sawyer*) and common noun phrases (*the boy*), the coref annotations also cover personal pronouns (*he*) as well.

We annotate three different categories of coreference phenomena -- coreference of identity (which links a mention in text to a discourse entity); copula (which links an attribute mention to another mention); and apposition (which links an appositive expression to another mention).

|Phenomenon|Example|Source|
|---|---|---|
|Coreference|One may as well begin with [Helen]<sub>x</sub>'s letters to [[her]<sub>x</sub> sister]<sub>y</sub>|Forster, *Howard's End*
|Copula|<img src="img/copula.png" alt="drawing" width="200"/>|Melville, *Bartleby, the Scrivener*|
| Apposition |<img src="img/apposition.png" alt="drawing" width="350"/>|Conrad, *Heart of Darkness*|

Annotations largely follow OntoNotes guidelines (though singleton mentions *are* annotated here), with several departures for literary style (including the distinction between generic/specific mentions, near-identity and the revelation of identity). For more on the coreference criteria used in these annotations, see David Bamman, Olivia Lewke and Anya Mansoor (2020), "[An Annotated Dataset of Coreference in English Literature](https://arxiv.org/abs/1912.01140)", LREC.

## Quotation annotations

The quotation layer in LitBank identifies all instances of direct speech in the text, attributed to its speaker.


|Quote|Speaker|Source|
|---|---|---|
— Come up , Kinch ! Come up , you fearful jesuit !|Buck\_Mulligan-0|Joyce, *Ulysses*|
‘ Oh dear ! Oh dear ! I shall be late ! ’|The\_White\_Rabbit-4|Carroll, *Alice in Wonderland*|
“ Do n't put your feet up there , Huckleberry ; ”|Miss\_Watson-26|Twain, *Huckleberry Finn*|

This layer captures dialogue in quotation marks and other typographical markers (such as dashes in Joyce's *Ulysses*), and excludes strings wrapped in quotation marks that do not constitute dialogue, such as the use of scare quotes for emphatic use (for jargon, neologisms or irony), titles of works of art, and the mention of a term (as distinct from its use).

Speaker labels are identical to those in the `coref/` section of LitBank, to enable linking the quotation and coreference layers.

For more on the quotation annotations, see [this paper](https://arxiv.org/pdf/2004.13980.pdf).

## Tagger

A trained tagger (which can be used to tag entities and events in new text) can be found in the `tagger/` directory.  (A model for coreference resolution is coming shortly.)

## Corpus

The corpus is drawn from the public domain texts on Project Gutenberg, and includes individual works of fiction (both novels and short stories) that include a mix of high literary style (e.g., Edith Wharton's *Age of Innocence*, James Joyce's *Ulysses*) and popular pulp fiction (e.g., H. Rider Haggard's *King Solomon's Mines*, Horatio Alger's *Ragged Dick*).  We select approximately 2,000 words from each of 100 texts; the total annotated dataset contains 210,532 tokens.


|Gutenberg ID|Date|Author|Title|
|---|---|---|---|
|514|1868|Alcott, Louisa May|Little Women|
|18581|1904|Alger, Horatio, Jr.|Adrift in New York: Tom and Florence Braving the World|
|5348|1868|Alger, Horatio, Jr.|Ragged Dick, Or, Street Life in New York with the Boot-Blacks|
|158|1815|Austen, Jane|Emma|
|105|1818|Austen, Jane|Persuasion|
|1342|1813|Austen, Jane|Pride and Prejudice|
|1206|1914|Bower, B. M.|The Flying U Ranch|
|969|1848|Brontë, Anne|The Tenant of Wildfell Hall|
|1260|1847|Brontë, Charlotte|Jane Eyre: An Autobiography|
|768|1847|Brontë, Emily|Wuthering Heights|
|2095|1853|Brown, William Wells|Clotelle: A Tale of the Southern States|
|113|1911|Burnett, Frances Hodgson|The Secret Garden|
|6053|1778|Burney, Fanny|Evelina, Or, the History of a Young Lady's Entrance into the World|
|62|1912|Burroughs, Edgar Rice|A Princess of Mars|
|78|1912|Burroughs, Edgar Rice|Tarzan of the Apes|
|2084|1903|Butler, Samuel|The Way of All Flesh|
|11|1865|Carroll, Lewis|Alice's Adventures in Wonderland|
|24|1913|Cather, Willa|O Pioneers!|
|44|1915|Cather, Willa|The Song of the Lark|
|472|1900|Chesnutt, Charles W. (Charles Waddell)|The House Behind the Cedars|
|1695|1908|Chesterton, G. K. (Gilbert Keith)|The Man Who Was Thursday: A Nightmare|
|160|1899|Chopin, Kate|The Awakening, and Selected Short Stories|
|1155|1922|Christie, Agatha|The Secret Adversary|
|155|1868|Collins, Wilkie|The Moonstone|
|219|1899|Conrad, Joseph|Heart of Darkness|
|974|1907|Conrad, Joseph|The Secret Agent: A Simple Tale|
|940|1826|Cooper, James Fenimore|The Last of the Mohicans; A narrative of 1757|
|73|1895|Crane, Stephen|The Red Badge of Courage: An Episode of the American Civil War|
|876|1861|Davis, Rebecca Harding|Life in the Iron-Mills; Or, The Korl Woman|
|521|1719|Defoe, Daniel|The Life and Adventures of Robinson Crusoe|
|1023|1852|Dickens, Charles|Bleak House|
|766|1849|Dickens, Charles|David Copperfield|
|1400|1861|Dickens, Charles|Great Expectations|
|730|1838|Dickens, Charles|Oliver Twist|
|1661|1892|Doyle, Arthur Conan|The Adventures of Sherlock Holmes|
|2852|1902|Doyle, Arthur Conan|The Hound of the Baskervilles|
|233|1900|Dreiser, Theodore|Sister Carrie: A Novel|
|15265|1911|Du Bois, W. E. B. (William Edward Burghardt)|The Quest of the Silver Fleece: A Novel|
|145|1871|Eliot, George|Middlemarch|
|550|1861|Eliot, George|Silas Marner|
|12677|1914|Ferber, Edna|Personality Plus: Some Experiences of Emma McChesney and Her Son, Jock|
|6593|1749|Fielding, Henry|History of Tom Jones, a Foundling|
|9830|1922|Fitzgerald, F. Scott (Francis Scott)|The Beautiful and Damned|
|805|1920|Fitzgerald, F. Scott (Francis Scott)|This Side of Paradise|
|2775|1915|Ford, Ford Madox|The Good Soldier|
|2641|1908|Forster, E. M. (Edward Morgan)|A Room with a View|
|2891|1910|Forster, E. M. (Edward Morgan)|Howards End|
|4276|1855|Gaskell, Elizabeth Cleghorn|North and South|
|32|1915|Gilman, Charlotte Perkins|Herland|
|502|1913|Grey, Zane|Desert Gold|
|3457|1919|Grey, Zane|The Man of the Forest|
|711|1887|Haggard, H. Rider (Henry Rider)|Allan Quatermain|
|2166|1885|Haggard, H. Rider (Henry Rider)|King Solomon's Mines|
|27|1874|Hardy, Thomas|Far from the Madding Crowd|
|110|1891|Hardy, Thomas|Tess of the d'Urbervilles: A Pure Woman|
|77|1851|Hawthorne, Nathaniel|The House of the Seven Gables|
|33|1850|Hawthorne, Nathaniel|The Scarlet Letter|
|95|1894|Hope, Anthony|The Prisoner of Zenda|
|41|1820|Irving, Washington|The Legend of Sleepy Hollow|
|208|1879|James, Henry|Daisy Miller: A Study|
|432|1903|James, Henry|The Ambassadors|
|209|1898|James, Henry|The Turn of the Screw|
|367|1896|Jewett, Sarah Orne|The Country of the Pointed Firs|
|2807|1899|Johnston, Mary|To Have and to Hold|
|4217|1916|Joyce, James|A Portrait of the Artist as a Young Man|
|2814|1914|Joyce, James|Dubliners|
|4300|1922|Joyce, James|Ulysses|
|217|1913|Lawrence, D. H. (David Herbert)|Sons and Lovers|
|543|1920|Lewis, Sinclair|Main Street|
|215|1903|London, Jack|The Call of the Wild|
|351|1915|Maugham, W. Somerset (William Somerset)|Of Human Bondage|
|11231|1853|Melville, Herman|Bartleby, the Scrivener: A Story of Wall-Street|
|2489|1851|Melville, Herman|Moby Dick; Or, The Whale|
|45|1908|Montgomery, L. M. (Lucy Maud)|Anne of Green Gables|
|41286|1866|Oliphant, Mrs. (Margaret)|Miss Marjoribanks|
|60|1905|Orczy, Emmuska Orczy, Baroness|The Scarlet Pimpernel|
|932|1839|Poe, Edgar Allan|The Fall of the House of Usher|
|1064|1842|Poe, Edgar Allan|The Masque of the Red Death|
|4051|1915|Praed, Campbell, Mrs.|Lady Bridget in the Never-Never Land: a story of Australian life|
|3268|1794|Radcliffe, Ann Ward|The Mysteries of Udolpho|
|434|1908|Rinehart, Mary Roberts|The Circular Staircase|
|171|1791|Rowson, Mrs.|Charlotte Temple|
|271|1877|Sewell, Anna|Black Beauty|
|84|1823|Shelley, Mary Wollstonecraft|Frankenstein; Or, The Modern Prometheus|
|120|1883|Stevenson, Robert Louis|Treasure Island|
|345|1897|Stoker, Bram|Dracula|
|829|1726|Swift, Jonathan|Gulliver's Travels into Several Remote Nations of the World|
|8867|1918|Tarkington, Booth|The Magnificent Ambersons|
|599|1848|Thackeray, William Makepeace|Vanity Fair|
|76|1884|Twain, Mark|Adventures of Huckleberry Finn|
|74|1876|Twain, Mark|The Adventures of Tom Sawyer|
|1327|1898|Von Arnim, Elizabeth|Elizabeth and Her German Garden|
|238|1915|Webster, Jean|Dear Enemy|
|5230|1897|Wells, H. G. (Herbert George)|The Invisible Man: A Grotesque Romance|
|36|1897|Wells, H. G. (Herbert George)|The War of the Worlds|
|541|1920|Wharton, Edith|The Age of Innocence|
|174|1890|Wilde, Oscar|The Picture of Dorian Gray|
|2005|1919|Wodehouse, P. G. (Pelham Grenville)|Piccadilly Jim|
|16357|1788|Wollstonecraft, Mary|Mary: A Fiction|
|1245|1919|Woolf, Virginia|Night and Day|

## Format

The entity data is formatted in the original [brat](http://brat.nlplab.org) standoff annotation format and in the tab-separated layered format of [https://github.com/meizhiju/layered-bilstm-crf](https://github.com/meizhiju/layered-bilstm-crf).

## Commons

We welcome contributions to LitBank in the form of annotations for new texts in the public domain and suggestions for new texts to include; please contact dbamman@berkeley.edu to get involved.  For more information:

* See [interannotator.com](http://www.interannotator.com/public/#/litbank/entities/155_the_moonstone_brat) to explore the current entity annotations in LitBank through the brat annotation interface.
* Try annotating entities in [O Pioneers!](http://www.interannotator.com/private/index.xhtml#/commons/test/24_o_pioneers_brat) yourself (login by hovering over the "brat" icon in the upper right corner, using username "test" and password "annotate").

## Acknowledgments

This work was supported by an Amazon Research Award ("Natural Language Processing for Literary Texts").

