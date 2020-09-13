import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gensim.models import KeyedVectors
from torchtext import data, vocab
import time

corp_path = r"data\nl"
corp_name = "nl_alpino-ud"
VECS_DIM = 256

bin_path = r"data\nl\nl_small\w2v_d256_ws5_lr0.01_ns0_iter4.bin"
word_vectors = KeyedVectors.load_word2vec_format(bin_path, binary=True)
vecs_tensor = torch.from_numpy(word_vectors.vectors)

TEXT = data.Field()
TAGS = data.Field(is_target=True)

train, val, test = data.TabularDataset.splits(
        path=corp_path, train='{}-train.tsv'.format(corp_name),
        validation ='{}-dev.tsv'.format(corp_name), 
        test='{}-test.tsv'.format(corp_name), format='tsv',
        fields=[('sentence', TEXT), ('tags', TAGS)])

TEXT.build_vocab(train)
TAGS.build_vocab(train)
TEXT.vocab.set_vectors(TEXT.vocab.stoi, vecs_tensor, VECS_DIM) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

train_iter, val_iter, test_iter = data.BucketIterator.splits(
         (train, val , test), batch_size=BATCH_SIZE,
         sort_key=lambda x: len(x.sentence), device=device)


class POSTaggerbiLSTM(nn.Module):
    def __init__(self, 
                 input_dim,
                 embedding_dim,
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 dropout, 
                 pad_idx):
        
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight = nn.Parameter(TEXT.vocab.vectors, requires_grad=False) # initilize weights
        
        self.l1 = nn.Linear(embedding_dim, hidden_dim)
        
        self.biLSTM = nn.LSTM(hidden_dim, 
                              hidden_dim, 
                              num_layers=n_layers, 
                              bidirectional=True, 
                              dropout=dropout if n_layers > 1 else 0)
        
        self.l2 = nn.Linear(hidden_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)        
        
    def forward(self, input_tensor):
        embedded = self.dropout(self.embedding(input_tensor)) 
#         projection = self.dropout(self.l1(embedded)) # size mismatch when passed to self.biLSTM
        outputs, (hidden, cell) = self.biLSTM(embedded)
        output_tensor = self.l2(self.dropout(outputs))
        mask = (input_tensor == PAD_IDX).unsqueeze(-1)
        output_tensor_predictions = F.log_softmax(output_tensor.masked_fill(mask, -np.inf))
        return output_tensor_predictions


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(TAGS.vocab)
N_LAYERS = 2
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = POSTaggerbiLSTM(INPUT_DIM, 
                        EMBEDDING_DIM, 
                        HIDDEN_DIM, 
                        OUTPUT_DIM, 
                        N_LAYERS, 
                        DROPOUT, 
                        PAD_IDX)

model.embedding.weight.data[PAD_IDX] = torch.zeros(256) # initialize the embedding of the pad token to zeros
TAG_PAD_IDX = TAGS.vocab.stoi[TAGS.pad_token]

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)                      
criterion = nn.NLLLoss(ignore_index=TAG_PAD_IDX)

model = model.to(device)
criterion = criterion.to(device)

    
def categorical_accuracy(preds, y, tag_pad_idx):
    """ Returns accuracy per batch """
    
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])
                     
    
def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.sentence)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = batch.tags.view(-1)
        
        loss = criterion(predictions, tags)
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.sentence)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = batch.tags.view(-1)
            
            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)


def epoch_time(start_time, end_time):
    """ Returns time taken for one epoch. """
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
best_valid_loss = float('inf')
writer = SummaryWriter()

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iter, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion, TAG_PAD_IDX)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'POStagger-model.pt')
        
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/dev', valid_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/dev', valid_acc, epoch)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    
# model.load_state_dict(torch.load('POStagger-model.pt'))
# test_loss, test_acc = evaluate(model, test_iter, criterion, TAG_PAD_IDX)

# print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')