import multiprocessing
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from time import time
import csv
from os import mkdir

lang = "de"
corpus_names = ["de_small"]
bats_sections = ['I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'I09', 'I10',  
                      'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10',
                      'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10',
                      'L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'L07', 'L08', 'L09', 'L10', 'Tot'] 
analogy_file = r'data\{}\{}_analogy.txt'.format(lang, lang)
bats_section_to_idx = {section: idx for idx, section in enumerate(bats_sections)} # map section names to integers


def write_csv(corpus_name, model_name):
    """Sets up a csv file to log intrinsic evaluation results."""
    
    with open(r"data\{}\{}\{}_analogy.csv".format(lang, corpus_name, model_name), 'w', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['epoch'] + bats_sections)
        
        
def train_word2vec(sentences, window, dim, alpha, neg_sampl, epoch_logger):  
    """Trains a word2vec model and saves the resulting embeddings at each epoch."""
    
    print("window size: {}, dimension: {}, learning rate: {}, negative sampling: {}".format(window, dim, alpha, neg_sampl))
    t = time()
    w2v_model = Word2Vec(sentences,
                    iter=5, # number of epochs
                    min_count=20,
                    window=window, # 
                    size=dim, # (256, 512)
                    sample=1e-5, # threshold at which higher-frequency words are randomly downsampled (0, 1e-5)
#                     sg=1, # Training algorithm: 1 for skip-gram; otherwise CBOW.
                    hs=neg_sampl, # If 1, hierarchical softmax will be used for model training. 
                            # If 0, and negative is non-zero, negative sampling will be used.
                    negative=10, # noise words for negative sampling (usually btw. 5-20)
                    alpha=alpha,
                    callbacks=[epoch_logger],
                    seed=42)
    
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))    
    
    
def compute_analogy_accuracies(wv, analogy_file):
    """
    Computes and returns accuracies for each section of the BATS analogy dataset, along with the mean accuracy.
    Input: Word2VecKeyedVectors, path to analogy dataset
    Output: list of accuracies
    """
    
    with open(analogy_file, 'r') as f:
        accuracies = []
        section_results = None
        for line in f:
            if line.startswith(':'):
                if section_results:
                    previous_accuracy = round(sum(section_results) / len(section_results), 4)
                    accuracies.append((section[:3], previous_accuracy))
                    print(section, ':', previous_accuracy)
                section = line[2:].rstrip()
                section_results = []
                continue

            example = line.split()
            pos = example[1].split('/') + example[2].split('/')
            neg = example[0].split('/')
            answer = example[3].split('/')
            num_answer = len(answer)
            num_correct = 0
            try:
                for word, score in wv.most_similar(positive=pos, negative=neg):
                    if word in answer:
                        num_correct += 1        
                section_results.append(num_correct / num_answer)
            except KeyError:
                pass
        mean_accuracy = round(sum([score for section, score in accuracies]) / len(accuracies), 4)
        print('Mean accuracy:', mean_accuracy)
        accuracies.append(('Tot', mean_accuracy))
    return accuracies
        
    
class Corpus(): 
    """An interator that yields sentences (lists of str)."""
    def __init__(self, path):
        self.corpus_path = path
        
    def __iter__(self):
        with open(corpus_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.split()
                

class EpochSaver(CallbackAny2Vec):
    """Callback to evaluate model after each epoch."""

    def __init__(self, corpus_name, model_name):
        self.epoch = 1
        self.corpus_name = corpus_name
        self.model_name = model_name

    def on_epoch_end(self, model):
        print('epoch', self.epoch)
        model.wv.vectors_norm = None
        accuracies = compute_analogy_accuracies(model.wv, analogy_file) 
        with open(r"data\{}\{}\{}_analogy.csv".format(lang, self.corpus_name, self.model_name), 'a', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            row = [None for _ in range(len(bats_sections))] 
            for section, score in accuracies:
                row[bats_section_to_idx[section[:3]]] = score # include section's accuracy at corresponding csv row index
            writer.writerow([self.epoch] + row) 
#         output_path = "data\{}\{}\{}_iter{}.bin".format(lang, self.corpus_name, self.model_name, self.epoch)
#         model.wv.save_word2vec_format(output_path, binary=True)
        self.epoch += 1
                         
                        
window_sizes = [10, 20]
dimensions = [256, 512]
learning_rates = [0.01, 0.05]
negative_sampling = [0, 1]

                        
for corpus_name in corpus_names:
    print(corpus_name)
    try:
        mkdir(r"data\{}\{}".format(lang, corpus_name))
    except FileExistsError: 
        pass
    corpus_path = r"data\{}\{}.txt".format(lang, corpus_name)
    sentences = Corpus(corpus_path)  
    for window in window_sizes:
        for dim in dimensions:
            for alpha in learning_rates:
                for value in negative_sampling:
                    model_name = "w2v_d{}_ws{}_lr{}_ns{}".format(dim, window, alpha, value)
                    write_csv(corpus_name, model_name)
                    epoch_logger = EpochSaver(corpus_name, model_name)
                    train_word2vec(sentences, window, dim, alpha, value, epoch_logger)
