# Embedding Masked Tokens

The purpose of this short investigation is to figure out how the presence of masked tokens affects the BERT model's embedding of the whole sentence. This information can be used to implement a better way of masking words in the main project as currently we are only able to predict the first token from each target word.

## The Method

In order to assess the impact that masked tokens have on the overall embedding of a sentence we are going to do a few comparisons between unaltered texts and their versions with each word masked at a time. Calculating the cosine similarity between the embeddings of the 2 sentences is a good measure of how much impact the new token has had on the sentence. Moreover, in order to evaluate how much the masked token affects the embedding for the rest of the sentence, we will also calculate embeddings on the sections of the sentence that do not contain the target word. For example, for the sentence "He went to the store", modified as "He went to the <MASKED>", we will calculate the cosine similarity between the embeddings of the "He went to the" section on each of the 2 sentences.

## The Test Inputs

We have generated 20 random sentences that vary in length and also contain a variety of words, including those more likely to be made up by multiple tokens (hyphenated composite words, conjugated verbs, etc.). We then go through every sentence at a time and then mask every word inside it in order to determine the cosine similarities between the original sentence and the masked sentence on both the full text and then also excluding the target word. These results are stored in the similarities.csv file.

## The Results

By calculating the average similarities inside of each sentence we can deduce some interesting information. There is a strong positive correlation (0.9) between the average full sentence similarity and the one excluding the target word, suggesting that, while the presence of a masked token affects the embedding for other tokens inside the sentence, the resulting effect is not very strong. The correlation between the sentence similarity and the length of the text is also quite high (0.74), meaning that the longer a sentence is, the less of an impact each masked token has.

Analysing the values associated with each word individually also reveals some details about the way in which BERT encodes these masked tokens. Below we have the table that compares the similarities that different kind of words produce after being masked inside their target sentence.

types | full sentence similarity	| similarity excluding current word
|--------|---------------------------|-------------------------------|
all words |	0.846 |	0.861
composite words |	0.853 |	0.928
non-composite words |	0.845 |	0.854
short words |	0.97 |	0.976
long words |	0.672 |	0.7

We can see that the impact of the masking of each word is correlated with the number of tokens, as we expected, but that composite words are less likely to have more tokens compared to longer words (the ones with more than 6 characters). This also suggests that our current method of only replacing the first token in each word should work about as well as a more complicated substitution, since the embeddings including the masked word are pretty similar to the ones excluding it.




