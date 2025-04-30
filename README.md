# anki deck generator 
this was created for learning Spanish, however it can be applied to other languages easily.

first run dl_freq.py to download the frequency dictionary from wiktionary. update this code (or the URL) to create the list of words


then run mk_ai_deck.py this will create a csv with translations to English with example sentences


then run gen.py  this will generate the audio for the word, examples, and photos for the image. you will need an openai API key. to use the latest image generation model you will need to have your account approved. 

be careful this is very expensive!

gen.py also creates the apkg file to be imported into anki. which i have provided in the releases section.

