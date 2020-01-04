#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/python
# -*- coding: utf-8 -*-

from io import open
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from pypinyin import pinyin, lazy_pinyin, Style
import time
# Record time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs 

# Train the sequence to sequence model
def train(device,input_tensor, target_tensor,dec_in, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,vocab_size, max_length,hidden_size,longueur):

    batch_size = input_tensor.size(0)
    encoder_hidden = torch.zeros(2, batch_size, hidden_size)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = input_tensor.size(0)
    target_length = dec_in.size(0) 
    
    encoder_outputs, encoder_hidden = encoder(input_tensor.to(device), encoder_hidden.to(device),longueur)
    
    decoder_hidden = encoder_hidden.to(device)
    decoder_outputs = torch.tensor(()).to(device)

    for j in range (0,target_length):
        decoder_input = dec_in[j]
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden.to(device), encoder_outputs.to(device))
        # Prepare final output
        decoder_outputs = torch.cat((decoder_outputs, decoder_output.to(device)),1)
        
    loss += criterion(decoder_outputs.view(decoder_outputs.shape[0]*decoder_outputs.shape[1],decoder_outputs.shape[2]), target_tensor.view(-1).to(device))
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() 


def eval_model(device,input_tensor, target_tensor,dec_in, encoder, decoder,criterion,vocab_size, max_length,hidden_size,int2char_dict,ep,longueur_valid):
    encoder.eval()
    decoder.eval()
    loss = 0
    batch_size = input_tensor.size(0)

    with torch.no_grad():             
        target_length = target_tensor.size(1)  
        # Encoder
        encoder_hidden =torch.zeros(2, batch_size, hidden_size)
        encoder_outputs, encoder_hidden = encoder(input_tensor.to(device), encoder_hidden.to(device),longueur_valid)

        # Decoder
        decoder_input = dec_in.to(device)
        decoder_hidden = encoder_hidden.to(device)
        decoder_outputs = torch.tensor((), device=device)
        # Prepare for model answer
        decoded_words = []  
        for k in range(batch_size):
            decoded_words.append('')
        for j in range (0,target_length): 
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden.to(device), encoder_outputs.to(device))
            topv, topi = decoder_output.data.topk(1)
            # Prepare next decoder input
            decoder_input = topi.squeeze().detach() 
            decoder_outputs = torch.cat((decoder_outputs, decoder_output.to(device)),1)

            for i in range(topv.shape[0]):
                if( j < longueur_valid[i]):
                    if(topi[i][0] == 1 or topi[i][0] ==2 or topi[i][0] ==0 or topi[i][0] == 3  ):
                        # do not break, but append top idx
                        decoded_words[i] = decoded_words[i] + str(topi[i][0].item())
                    else:
                        decoded_words[i] =decoded_words[i] +str(int2char_dict.get(topi[i][0].item()))

        # Prepare evalutation results
        decoded_words = '\n'.join(decoded_words)
        path = "./"+str(ep)+"_"+str(hidden_size)+"_eval_model_.txt"
        write_to(path, decoded_words)
        if(ep == 0):
            write_to_out("./"+str(ep)+"_"+str(hidden_size)+"_eval_out_.txt", detokenize_All(target_tensor,int2char_dict ))
            write_to_out("./"+str(ep)+"_"+str(hidden_size)+"_eval_in_.txt", detokenize_All(input_tensor,int2char_dict ))

        # Compute loss
        loss += criterion(decoder_outputs.view(decoder_outputs.shape[0]*decoder_outputs.shape[1],decoder_outputs.shape[2]), target_tensor.view(-1).to(device))

    return loss.item()

# Write the evalutation result
def write_to(path,words):
    f = open(path,"w")
    f.write(words)
    f.write('\n')
    f.write("###############################")
    f.write('\n')
    f.close
# Write the evalutation result
def write_to_out(path,words):
    f = open(path,"w")
    f.write(words)
    f.write("###############################")
    f.write('\n')
    f.close

# Strat training and evaluating the model
def trainCouplets(device,encoder, decoder,batch_in_out_pairs, vocab_size,max_length,hidden_size, encoder_optimizer, decoder_optimizer,int2char_dict,indices_train,indices_valid,ep,lon):

    start = time.time()
    plot_losses = []
    train_loss = 0 
    valid_loss = 0  
    
    if (torch.cuda.is_available()):
        encoder.cuda()
        decoder.cuda()
    encoder.train()
    decoder.train()
    # Define loss
    criterion = nn.CrossEntropyLoss(ignore_index = 0)

    np.random.shuffle(indices_train)
    # Training
    for idx in range( len(indices_train)):
        idx_couplet_batch = indices_train[idx]
        training_pair = batch_in_out_pairs[idx_couplet_batch]
        longueur = lon[idx_couplet_batch]

        dec_in = torch.transpose(training_pair[1], 1,0)
        input_tensor = training_pair[0]
        target_tensor = training_pair[2]

        loss_t = train(device,input_tensor, target_tensor,dec_in, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,vocab_size, max_length,hidden_size,torch.tensor(longueur).to(device))
        train_loss += loss_t
        
    # Evaluating
    for idx in range( len(indices_valid)):
        idx_couplet_batch = indices_valid[idx]
        validation_pair = batch_in_out_pairs[idx_couplet_batch]
        longueur_valid = lon[idx_couplet_batch]

        dec_in_valid = torch.transpose(batch_in_out_pairs[0][1],0,1)[0] # only 1's
        input_tensor_valid = validation_pair[0]
        target_tensor_valid = validation_pair[2]  

        loss_v = eval_model(device, input_tensor_valid, target_tensor_valid,dec_in_valid, encoder,decoder,criterion,vocab_size, max_length,hidden_size,int2char_dict,ep,torch.tensor(longueur_valid).to(device))
        valid_loss += loss_v
            
    return train_loss/len(indices_train) , valid_loss / len(indices_valid)


# Load vocabulary from a txt file
def load_vocab(file_path):
    # load vocab.txt with \n at each line
    f = open(file_path,"r")
    lines = f.readlines()
    
    vocab = data_clean(lines)
    return vocab

# Load couplets data
def load_data(in_path, out_path, nb_couplets,with_all_couplets):
    couplet_in = open(in_path, "r")
    couplet_out = open(out_path, "r")

    # load original couplets
    lines_in = couplet_in.readlines() 
    lines_out = couplet_out.readlines()

    # clean samples
    samples_in = data_clean(lines_in)
    samples_out = data_clean(lines_out)

    # determine min and max length
    min_length = len(min(samples_in, key=len))
    max_length = len(max(samples_out, key=len))
    max_length +=  1
    print("min line length is ", min_length)
    print("max line length is ", max_length)
    if (with_all_couplets == False):
        samples_in = samples_in[:nb_couplets]  # take some samples
        samples_out = samples_out[:nb_couplets]

    print("Loaded input couplets length is ",len(samples_in))
    print("Loaded output couplets length is ",len(samples_out))

    return samples_in, samples_out, max_length


# clean data
def data_clean(data):
    s=[]
    for i in range(0, len(data)):
        s.append(str(data[i]).replace('\n','').replace(' ','')) 
    return s


# create char2int dictionary
def create_char2int_and_int2char_dict(vocab):
    rang = np.arange(len(vocab))
    char2int_dict = dict(zip(vocab,rang))
    int2char_dict = dict(zip(rang,vocab))
    int2char_dict[0] = ''  # '' for pad
    return char2int_dict,int2char_dict


# create char2tone dictionary
def create_char2tone_dict(vocab):
    tone_dict = dict()
    tone_dict[vocab[0]]= ' '
    tone_dict[vocab[1]]= ' '
    tone_dict[vocab[2]]= ' '
    tone_dict[vocab[3]]= ' '
    for i in range(4,len(vocab)): 
        key = str(vocab[i])
        t = pinyin(key, style=Style.TONE3, heteronym=True)[0] # tones for the character
        for j in range(len(t)):
            
            pz = str(t[j][-1])  # tone number
            if ( pz!= (key) and pz.isdigit() ):  
                if( int(pz) <= 2):
                    if (j==0):
                        tone_dict[key]=['P']
                    else:
                        tone_dict[key].append('P')  
                else:
                    if(j==0):
                        tone_dict[key]=['Z']
                    else:
                        tone_dict[key].append('Z')
            else: 
                if (tone_dict.get(key) == None):
                    tone_dict[key]=[' ']  # when there is no tone  
                
        tone_dict[key] = set(tone_dict[key])
        tone_dict[key] = list(tone_dict[key])
        res =''
        for k in range(len(tone_dict[key])):
            res += tone_dict[key][k]
        tone_dict[key] = res
    return tone_dict


    
# tokenize: couplet to int vectors
def tokenize(data, char2int_dict, max_length): 
    v=[]
    lon=[]
    for i in range(0, len(data)):
        l = []
        lon.append(len(data[i]))
        for j in range (0, len(data[i])):
            if ( char2int_dict.get(data[i][j])  is None ):
                l.append(   char2int_dict.get('UNK')  )
            else:
                l.append(   char2int_dict.get(data[i][j])  )
        v.append(torch.tensor(l, dtype=torch.long).view(-1, 1))
    return v,lon


def tokenize_dec_in(data, char2int_dict, max_length):  
    v=[]
    for i in range(0, len(data)):
        l = [char2int_dict.get('sos')]

        for j in range (0, len(data[i])):
            if ( char2int_dict.get(data[i][j])  is None ):
                l.append(   char2int_dict.get('UNK')  )
            else:
                l.append(   char2int_dict.get(data[i][j])  )
        v.append(torch.tensor(l, dtype=torch.long).view(-1, 1))
    return v


def tokenize_dec_out(data, char2int_dict, max_length): 
    v=[]
    for i in range(0, len(data)):
        l = []
        for j in range (0, len(data[i])):
            if ( char2int_dict.get(data[i][j])  is None ):
                l.append(   char2int_dict.get('UNK')  )
            else:
                l.append(   char2int_dict.get(data[i][j])  )

        l.append(char2int_dict.get('eos'))
        v.append(torch.tensor(l, dtype=torch.long).view(-1, 1))
    return v

# Convert indices to characters
def detokenize(one_couplet, int2char_dict):
    o = ''
    for i in range(0, len(one_couplet)):
        if (one_couplet[i] >= 3):    # ignore pad,sos,eos
            o = o + int2char_dict.get(one_couplet[i].item())  
    return o
    
# Convert indices to characters
def detokenize_II(one_couplet, int2char_dict,n):
    o = ''
    n = min(len(one_couplet),n)
    for i in range(0, n):
        if (couplet[i] >= 3):    # ignore pad,sos,eos
            o = o + int2char_dict.get(one_couplet[i].item())  

    return o
# Convert indices to characters
def detokenize_All(couplets, int2char_dict):
    o = ''
    for i in range(0, len(couplets)):
        for j in range(len(couplets[i])):
            if (couplets[i][j] >= 3):    # ignore pad,sos,eos
                o = o + int2char_dict.get(couplets[i][j].item()) 
        o = o +"\n"
    return o


# create a dictionary of repeated words
def dict_repet_idx(one_tensor,length):
    records_array = np.array(one_tensor)[:length]
    vals, inverse, count = np.unique(records_array, return_inverse=True,return_counts=True)
    idx_vals_repeated = np.where(count > 1)[0]
    vals_repeated = vals[idx_vals_repeated]

    rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
    _, inverse_rows = np.unique(rows, return_index=True)
    res = np.split(cols, inverse_rows[1:])
    
    d = dict()
    for i in range(len(res)):
        for l in range(1,len(res[i])):
            val = res[i][0]
            d[ res[i][l] ]  = val
    
    return d
