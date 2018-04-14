import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
np.random.seed(50000)

#global
words = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
num_topics = 3


class Generator:

    def __init__(self,alpha=0.1,beta=0.01):
        self.alpha =alpha
        self.beta = beta

    def generate_documents(self):
        documents = []
        word_topic_dist = np.random.dirichlet([self.beta]*20, size=3)
        topic_generated = np.random.dirichlet([self.alpha]*3, size=200)
        for i in topic_generated:
            actual_doc = ''
            for j in range(0, 50):
                topic_distribution = np.random.multinomial(1, i, size=1)
                topic = np.argmax(topic_distribution)
                word_picked = np.random.multinomial(1, word_topic_dist[topic], size=1)
                word = int(np.argmax(word_picked))
                actual_doc = actual_doc + words[word] + " "
            documents.append(actual_doc)
        #print(documents)
        return documents, word_topic_dist, topic_generated


class Utils:
    @staticmethod
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % topic_idx)
            print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    @staticmethod
    def compare_distributions(true_dist, recovered_dist):
        for i in range(0, 3):
            x = np.arange(20)
            plt.xticks(x, words)
            plt.xlabel('labels')
            plt.ylabel('P(W|T)')
            plt.title('Word-Topic distribution for all the topics')
            plt.plot(true_dist[i], color='green')
            plt.plot(recovered_dist[i], color='red')
            plt.show()
            from scipy import stats
            print(i, i, stats.entropy(recovered_dist[i], true_dist[i]))

    @staticmethod
    def compute_entropy(topic_document):
        sum_entropy = 0
        for i in topic_document:
            temp = 0
            for j in i:
                if j != 0:
                    temp = temp + j * math.log(j)
            sum_entropy = sum_entropy + temp
        mean_entropy = -sum_entropy / len(topic_document)
        return mean_entropy


class LDA:
    def __init__(self,documents,alpha,beta):
        self.documents = documents
        self.alpha = alpha
        self.beta = beta
        self.lda = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=alpha, topic_word_prior=beta)
        self.count_vec = CountVectorizer(stop_words=None, analyzer='char', lowercase=False, max_df=0.99)

    def computeLDA(self):
        tf = self.count_vec.fit_transform(self.documents)
        tf_feature_names = self.count_vec.get_feature_names()
        self.lda.fit(tf)
        topic_given_document = self.lda.transform(X=tf)

        no_top_words = 10
        Utils.display_topics(self.lda, tf_feature_names, no_top_words)
        lda_prob = self.lda.components_ / self.lda.components_.sum(axis=1)[:, np.newaxis]
        lda_pr = np.zeros((3, 20))
        for i, topic in enumerate(lda_prob):
            for j, word in enumerate(tf_feature_names):
                true_idx = words.index(word)
                lda_pr[i][true_idx] = topic[j]
        return lda_pr, topic_given_document


def generate_distribution_for_alpha(alphas):
    mean_vals = OrderedDict()
    gen = Generator(0.1, 0.01)
    docs, word_topic_dist, topic_generated = gen.generate_documents()
    mean = Utils.compute_entropy(topic_generated)
    for alpha in alphas:
        lda = LDA(docs, alpha, beta)
        p_lda, topic_list = lda.computeLDA()
        mean_vals[alpha] = Utils.compute_entropy(topic_list)
    plt.title("Mean entropy for generative LDA model is {}".format(mean))
    plt.xlabel("List of Alpha values")
    plt.ylabel("Mean Entropy of recovered LDA model")
    plt.plot(list(mean_vals.keys()), list(mean_vals.values()))
    plt.show()


def generate_distribution_for_beta(beta_list):
    mean_vals = OrderedDict()
    gen = Generator(0.1, 0.01)
    docs, word_topic_dist, _ = gen.generate_documents()
    mean = Utils.compute_entropy(word_topic_dist)
    for beta in beta_list:
        lda = LDA(docs, alpha, beta)
        p_lda, _ = lda.computeLDA()
        mean_vals[beta] = Utils.compute_entropy(p_lda)
    plt.title("Mean entropy for generative LDA model is {}".format(mean))
    plt.xlabel("List of Beta values")
    plt.ylabel("Mean Entropy of recovered LDA model")
    plt.plot(list(mean_vals.keys()), list(mean_vals.values()))
    plt.show()


if __name__ == "__main__":
    alpha = 0.1
    beta = 0.01
    # Part 1
    gen = Generator(alpha, beta)
    docs, word_given_topic, topic_generated = gen.generate_documents()
    print(word_given_topic.shape, topic_generated.shape)

    #sample docs
    print(docs[1])
    print(docs[10])

    #print topic distribution
    for i in range(len(words)):
        print(words[i], end=" ")
        for topic in range(len(word_given_topic)):
            print("%.4f"%word_given_topic[topic][i], end=' ')
        print()

    # Part 2
    model = LDA(docs, alpha,beta)
    recovered_word_topic, _ = model.computeLDA()
    Utils.compare_distributions(word_given_topic, recovered_word_topic)

    # Part 3
    list_alpha = [0.1, 1, 5, 10, 15, 20, 30, 40, 50]
    generate_distribution_for_alpha(list_alpha)

    list_beta = [0.1, 1, 20, 40, 60, 100]
    generate_distribution_for_beta(list_beta)
