# %% [markdown]
# Getting Started with Python's NLTK

# %%
import nltk
import ssl
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# %% [markdown]
# TOKENIZING

# %%
example_string = """
... Muad'Dib learned rapidly because his first training was in how to learn.
... And the first lesson of all was the basic trust that he could learn.
... It's shocking to find how many people do not believe they can learn,
... and how many more believe learning to be difficult."""
#seperate string into sentences. returns a list or array of sentences
sent_tokenize(example_string)



# %%
#seperate words in the string. returns a list or array of words.
#notice it's was broken into two words it and 's emphasizing it is
#notice Muad'Dib was not broken because it is not a possessive apostrophe
word_tokenize(example_string)

# %%
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from nltk.corpus import stopwords

# %%
worf_quote = "Sir, I protest. I am not a merry man!"
words_inquote = word_tokenize(worf_quote)
words_inquote

# %%
#filtered out stop words such as am, i, in, a. Words that will not affect the overall meaning of the sentence
stop_words = set(stopwords.words("english"))
filtered_list = []
for word in words_inquote:
    if word.casefold() not in stop_words:
        filtered_list.append(word)

filtered_list

# %% [markdown]
# STEMMING

# %%

stemmer = PorterStemmer()
string_for_stemming = "The crew of the USS Discovery discovered many discoveries. Discovering is what explorers do."
words_to_stem = word_tokenize(string_for_stemming)
words_to_stem

# %%
stemmed_words = [stemmer.stem(word) for word in words_to_stem]
stemmed_words
#todo: explore Porter2 - Snowbal stemmer

# %% [markdown]
# Tagging Parts of Speech

# %%
#words are broken down into tuples by parts of speech (word, pos_tag). kind of like a parser
nltk.download('averaged_perceptron_tagger_eng')
sagan_quote = """
 If you wish to make an apple pie from scratch,
 you must first invent the universe."""
words_in_sagan_quote = word_tokenize(sagan_quote)
nltk.pos_tag(words_in_sagan_quote)

# %%
nltk.download('tagsets_json')
nltk.help.upenn_tagset() #for parts of speech tags and their meanings

# %%
jabberwocky_excerpt = """'Twas brillig, and the slithy toves did gyre and gimble in the wabe:
all mimsy were the borogoves, and the mome raths outgrabe."""
words_in_excerpt = word_tokenize(jabberwocky_excerpt)
nltk.pos_tag(words_in_excerpt)

# %% [markdown]
# Lemmatizing

# %%
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("scarves")


# %%
string_for_lemmatizing = "The friends of DeSoto love scarves."
words = word_tokenize(string_for_lemmatizing)
words
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
lemmatized_words

# %%
lemmatizer.lemmatize("better", pos="a")


# %% [markdown]
# CHUNKING

# %%
lotr_quote = "It's a dangerous business, Frodo, going out your door."
words_in_lotr_quote = word_tokenize(lotr_quote)
words_in_lotr_quote
lotr_pos_tags = nltk.pos_tag(words_in_lotr_quote)
lotr_pos_tags
#chunk grammar sample. uses regex. DT stands for determiner | JJ for adjective || NN for Noun
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(lotr_pos_tags)
tree.draw()

# %% [markdown]
# CHINKING

# %%
grammar = "Chunk: {<.*>+}}<JJ>{"
tree2 = chunk_parser.parse(lotr_pos_tags)
tree2
tree2.draw()

# %% [markdown]
# Using Named Entity Recognition

# %%
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")
tree3 = nltk.ne_chunk(lotr_pos_tags)
tree.draw()

# %%
tree = nltk.ne_chunk(lotr_pos_tags, binary=True)
tree.draw()

# %%
quote = """
Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2."""

def extract_ne(quote):
    words = word_tokenize(quote, language="english")
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(" ".join(i[0] for i in t)
       for t in tree
       if hasattr(t, "label") and t.label() == "NE"
       )
extract_ne(quote)

# %% [markdown]
# Getting Text to Analyze

# %%
nltk.download("book")
from nltk.book import *

# %%
#concordance displays each time a word is used, along with its immediate context
text8.concordance("man")

# %%
text8.concordance("woman")

# %% [markdown]
# Making a Dispersion Plot

# %%
text8.dispersion_plot(["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"])

# %%
text2.dispersion_plot(["Allenham", "Whitwell", "Cleveland", "Combe"])

# %% [markdown]
# Making a Frequency Distribution

# %%
frequency_distribution = FreqDist(text8)
print(frequency_distribution)

# %%
frequency_distribution.most_common(20)

# %%
meaningful_words = [word for word in text8 if word.casefold() not in stop_words]
frequency_distribution = FreqDist(meaningful_words)
frequency_distribution.most_common(20)

# %%
frequency_distribution.plot(20, cumulative=True)

# %% [markdown]
# Finding Collocations

# %%
lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]
new_text = nltk.Text(lemmatized_words)
new_text.collocations()


