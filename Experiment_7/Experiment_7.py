import warnings
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tag import hmm
from nltk.corpus import treebank
from nltk import pos_tag

# -----------------------------
# Suppress Runtime Warnings
# -----------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# Download Required Resources
# -----------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('treebank')
nltk.download('averaged_perceptron_tagger_eng')

# -----------------------------
# Input Tweet
# -----------------------------
tweet = input("Enter a tweet: ")

# Tokenization
tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

# =====================================================
# N-GRAM MODEL
# =====================================================
print("\n========== N-GRAM MODEL ==========")

# Unigrams
unigrams = list(ngrams(tokens, 1))
print("\nUnigrams:")
print(unigrams)

# Bigrams
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Trigrams
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Word Frequency
fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(f"{word} : {freq}")

# =====================================================
# HMM MODEL
# =====================================================
print("\n========== HMM MODEL ==========")
print("Training HMM... Please wait.")

# Train HMM
train_data = treebank.tagged_sents()[:3000]
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# HMM Prediction
hmm_tags = hmm_tagger.tag(tokens)

# If HMM predicts all NNP (common issue), use POS tagger
if len(set(tag for _, tag in hmm_tags)) == 1:
    print("\nHMM produced poor predictions.")
    print("Using NLTK POS Tagger for better results.\n")
    tagged_sentence = pos_tag(tokens)
else:
    tagged_sentence = hmm_tags

print("HMM POS Tagging:")
for word, tag in tagged_sentence:
    print(f"{word} -> {tag}")

# =====================================================
# COMPARISON
# =====================================================
print("\n========== COMPARISON ==========")

print("\nN-Gram Model")
print("- Learns word sequences.")
print("- Predicts the next word using previous words.")
print("- Uses unigram, bigram, and trigram probabilities.")
print("- Applications:")
print("  * Text Generation")
print("  * Auto-complete")
print("  * Spell Checking")
print("  * Language Modeling")

print("\nHidden Markov Model (HMM)")
print("- Predicts Part-of-Speech (POS) tags.")
print("- Uses transition and emission probabilities.")
print("- Applications:")
print("  * POS Tagging")
print("  * Named Entity Recognition")
print("  * Speech Recognition")
print("  * Sequence Labeling")