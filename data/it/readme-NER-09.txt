NER-09.zip  contains the data  used for  the Named  Entity Recognition
(NER) Task at EVALITA 2009.

Data  are divided  into  a  development part  (which  consists of  the
whole dataset used for the NER task at EVALITA 2007) and a test part.

Development data  and test  data consist of  two separate  text files,
with one token per line and an empty line after each sentence.

Named Entities are annotated in the IOB2 format.

The Named Entity tag consists of two parts:
1. the  IOB2 tag: 'B'  (for 'begin')  denotes  the  first  token  of a
   Named Entity, I (for 'inside')  is used for  all other tokens  in a
   Named Entity, and 'O' (for 'outside') is used for all other words;
2. the Entity  type  tag: PER  (for Person), ORG  (for  Organization),
   GPE (for Geo- Political Entity), or LOC (for Location).

Both development and  test data have been further  annotated with Part
of Speech information1 using  the Elsnet tagset for Italian (available
at:  http://evalita.itc.it/tasks/elsnet-tagset-IT.pdf).  Please notice
that  the corpus  has  been PoS-tagged  automatically  with no  manual
correction.

Each file  consists of four  columns separated by a  blank, containing
respectively the  token, the Elsnet  PoS-tag, the Adige news  story to
which the token belongs, and the Named Entity tag.

Example:
il RS adige20041008_id414157 O
capitano SS adige20041008_id414157 O
della ES adige20041008_id414157 O
Gerolsteiner SPN adige20041008_id414157 B-ORG
Davide SPN adige20041008_id414157 B-PER
Rebellin SPN adige20041008_id414157 I-PER
