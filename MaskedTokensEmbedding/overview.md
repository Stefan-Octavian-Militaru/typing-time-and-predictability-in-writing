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

# A different approach

A different method of isolating the target words inside their respective sentence would be to surround them with 2 custom tokens, which we will call <START_WORD> and <END_WORD>. The main problem with the previous approach was the way in which BERT predicted our masked target word, being able to only look at its first token. This new method may help rectify this issue, however it also has its own dissadvantages such as the possibility that introducing 2 new tokens may alter the original sentence's embedding in a more drastic manner.

## The Method

Changing our code to reflect this new approach will not entail a massive restructuring, only a few alterations to the process of determining the modified sentence. We will also no longer need to calculate the similarity of the sentences excluding our target word like in the previous method, but instead we will compute the similarities between the embeddings of our isolated target word in both of the sentences.

## The results

As with the previous method, the analysis starts by looking at the average similarities of each sentence. Correlating the full sentence and target word only similarities gives us a value of 0.9, which is identical to the one observed with the other method, suggesting that the impact that tokens have in the embedding is uniform throughout the whole sentence. Looking at the correlation between sentence length and each type of similarity, we see that it hovers around 0 (0.05 and -0.15 respectively), which differs from the results observed with a single masked token. This drastic difference is likely due to this approach changing the total length of the sentence by substituting 1 word with 2 masked tokens.

Moving on, we can analyise our outputs by looking at the results for each individual word. Here we get a correlation of 0.97 between the full sentence similarity and target word only similarity, strengthening the observation that the impact of these custom tokens is spread out uniformly throughout the embedding of the whole sentence. Correlating the amount of tokens in each word with the 2 types of similarities we get the values of 0.39 and 0.42, which reveal that while the length of the word's representation withing the BERT model does affect the final results, the change in embeddings cannot be justified solely through it.

Here is the table of similarities calculated through this second method.
types | full sentence similarity	| target word only similarity
|--------|---------------------------|-------------------------------|
all words |	0.743 |	0.655
composite words |	0.805 |	0.736
non-composite words |	0.738 |	0.648
short words |	0.84 |	0.733
long words |	0.607 |	0.545

Comparing these results with the ones generated by the previous method, we deduce that this second approach is indeed prone to altering the sentence embedding in a more drastic manner. This fact is likely due to many factors, some of which we have detected in the analysis provided above, such as: the presence of 2 custom tokens instead of just 1, the increased length of the total sentence, the uniform change in embedding throughout the whole sentence as opposed to only around the modified area, etc.

The csv files that contain the outputs generated by the 2 approaches are included in this repositories, under the names of similarities_m1 and similarities_m2.
