# ----------- File For Reading Data-----------
import readTrainData as Rdata
# ----------- Python Packages -----------
from collections import Counter
import numpy as np


# A function that computes and returns the probabilities:
#   Pw - a matrix of class-conditional probabilities, p(x|wi)
#   P - a vector of class priors, p(wi)
def learn_NB_text():
    texAll, lbAll, voc, cat = Rdata.readTrainData("r8-train-stemmed.txt")
    cat = list(cat)
    cat.sort()
    voc = list(voc)
    voc.sort()

    pr_w = posterior(texAll, lbAll, voc, cat, alpha=1)  # p(Xj|Wi)
    p = calc_p(cat, lbAll)

    return pr_w, p


# Returns Pw - a matrix of class-conditional probabilities, p(x|wi)
def posterior(data: list, labeled_data: list, words: list, categories: list, alpha):

    pr_w = {}
    words_in_categories = [list()] * len(categories)  # List of all words in each category

    for i in range(len(categories)):  # For every category
        for j in range(len(labeled_data)):
            if categories[i] == labeled_data[j]:  # If the label of the sentence equal to category
                words_in_categories[i] = words_in_categories[i] + data[j]  # Add the words to the category list

    # Calc with laplace smoothing the probability for each class. alpha = 1
    for i, cat in enumerate(categories):
        count_words_in_cat = Counter(words_in_categories[i])
        pr_w[cat] = {word: (count_words_in_cat[words[j]] + alpha) / (len(words_in_categories[i]) + (alpha * len(words))) for j, word in enumerate(words)}
        pr_w[cat]["unknown_word"] = alpha / (len(words_in_categories[i]) + (alpha * len(cat)))

    return pr_w


# Returns P - a vector of class priors, p(wi)
def calc_p(categories: list, data: list):
    appearance = []
    word_counts = Counter(data)

    for word in categories:
        appearance.append(word_counts[word] / len(data))

    return appearance


# Function that classifies all documents from the test set and computes the success rate
# as a number of correctly classified documents divided by the number of all documents in the test set.
def ClassifyNB_text(Pw, P):
    texAll, lbAll, _, _ = Rdata.readTrainData("r8-test-stemmed.txt")
    _, _, _, cat = Rdata.readTrainData("r8-train-stemmed.txt")
    cat = list(cat)
    cat.sort()
    success = 0

    for index, sentence in enumerate(texAll):
        label_sentence = lbAll[index]

        # Sum the probability Pw[word|category] for each word in the sentence
        categories_mul = [sum(np.log(Pw[category][word]) for word in sentence if word in Pw[category]) +
                          sum(np.log(Pw[category]["unknown_word"]) for word in sentence if word not in Pw[category]) +
                          np.log(P[i])
                          for i, category in enumerate(cat)]

        # Finding the category with the max probability, and compare with the label of the current sentence
        max_labels_index = categories_mul.index(max(categories_mul))
        if cat[max_labels_index] == label_sentence:
            success += 1

    return success / len(texAll)


if __name__ == '__main__':
    Pw, P = learn_NB_text()
    success = ClassifyNB_text(Pw, P)
    print(success)
