#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import argparse
import math

from utils import *
from encoderRNN import EncoderRNN
from attnDecoderRNN_II import AttnDecoderRNN_II


# Main function
if __name__ == '__main__':
	parse = argparse.ArgumentParser()
	parse.add_argument("-bs", "--batch_size", type=int, default=256, help="batch size")
	parse.add_argument("-hs", "--hidden_size", type=int,required=False,default= 256, help="hidden size")

	parse.add_argument("-lr","--learning_rate", type=float, default=0.0001, help='learning rate')
	parse.add_argument("-ep",'--epoch', type=int, default=1, help='epoch')
	parse.add_argument("-ra",'--ratio', type=float, default=0.8, help='the size of the validation dataset depends on the ratio')
	parse.add_argument("-dr",'--dropout_p', type=float, default=0.1, help='the probability for the dropout layer in attnDecoder')

	parse.add_argument("-a",'--with_all_couplets', default = False, help='use all couplets to train')
	parse.add_argument("-nb",'--nb_couplets',type=int, default=500, help='number of couplets to train')

	# Load all parameters
	args = vars(parse.parse_args())
	batch_size = args['batch_size']
	hidden_size = args['hidden_size']
	learning_rate= args['learning_rate']
	epoch = args['epoch']
	ratio = args['ratio']
	with_all_couplets = args['with_all_couplets']
	nb_couplets = args['nb_couplets']
	dropout_p = args['dropout_p']

	print("### Hyperparameters ###")
	print("Batch size is: ",batch_size)
	print("Hidden size is: ",hidden_size)
	print("Learning rate is: ",learning_rate)
	print("Epoch is: ",epoch)
	print("Ratio is: ",ratio)
	print("Dropout_p is: ",dropout_p)

	print("Use all couplets to train: ",with_all_couplets)
	if (with_all_couplets==False):
		print("\tTherefore nb_couplets to train is: ",nb_couplets)
	print("### Data loading ### ")

	# Couplet path
	in_path = "./in_clean.txt"
	out_path = "./out_clean.txt"
	# Vocabulary path
	file_path = "./vocab_total_ori.txt"
	# Load and clean couplet data
	data_in, data_out, max_length = load_data(in_path,out_path,nb_couplets,with_all_couplets)
	# Load vocabulary
	vocab = load_vocab(file_path)
	vocab_size = len(vocab)
	print("Vocab length is ", vocab_size)

	print("### Data preparing ### ")
	# Create char2int dictionary and int2char dictionary for latter use
	char2int_dict, int2char_dict = create_char2int_and_int2char_dict(vocab)
	# Tokenize the input couplets and return a list containing lengths of each couplrt
	Enc_in,lon =  tokenize(data_in,char2int_dict, max_length)
	# Tokenize couplet output
	token_out =tokenize(data_out,char2int_dict, max_length)
	# Prepare for decoder input and ideal decoder output(target)
	Dec_in = tokenize_dec_in(data_out,char2int_dict, max_length)
	Dec_out = tokenize_dec_out(data_out,char2int_dict, max_length)  

	# Prepare padded batches
	# Encoder input
	in_ = []
	longueur=[]
	for i in range(0,int(len(Enc_in)/batch_size)):
		a = (Enc_in[batch_size*i:batch_size*(i+1)])
		# Prepare corresponding list of couplet lengths for latter use in Encoder
		longueur.append(lon[batch_size*i:batch_size*(i+1)] )
		# Pad sequence
		in_.append( torch.nn.utils.rnn.pad_sequence(a, batch_first=True) )
	# Ideal decoder output
	out_ = []
	for i in range(0,int(len(Dec_out)/batch_size)):
		a = (Dec_out[batch_size*i:batch_size*(i+1)])
		# Pad sequence
		out_.append( torch.nn.utils.rnn.pad_sequence(a, batch_first=True)   )
	# Decoder input
	dec_in_ = []
	for i in range(0,int(len(Dec_in)/batch_size)):
		a = (Dec_in[batch_size*i:batch_size*(i+1)])
		# Pad sequence
		dec_in_.append( torch.nn.utils.rnn.pad_sequence(a, batch_first=True)   )

	# Prepare [Encoder input,Decoder input,Ideal decoder output] set
	batch_in_out_pairs =[]   # may ignore some last couplets
	for i in range(0, len(in_)):
		batch_in_out_pairs.append((in_[i].squeeze(),dec_in_[i].squeeze(), out_[i].squeeze()))
	print("Number of couplet batches :",len(batch_in_out_pairs))

	### Starting Training ###
	print("### Creating models ###")
	# Define encoder and decoder
	input_size = vocab_size
	output_size = vocab_size
	encoder = EncoderRNN(input_size, hidden_size)
	attn_decoder = AttnDecoderRNN_II(hidden_size, output_size,dropout_p)
	# Define the device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoder.to(device)
	attn_decoder.to(device)

	# Define the optimizer
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=learning_rate)

	lossesPLL_train = []
	lossesPLL_valid = []
	times = []

	print("### Starting training ###")
	# Prepare indices for train set and validation set
	split = int(len(batch_in_out_pairs)*ratio)
	indices = np.arange(len(batch_in_out_pairs))
	np.random.shuffle(indices)
	indices_train = indices[:split]
	indices_valid = indices[split:]
	print("Train indices are:",indices_train)
	print("Valid indices are:",indices_valid)

	for ep in range(epoch):
		print("## Epoch ",ep," ## ")
		start_time = time.time()
		# Train couplets
		train_loss, valid_loss = trainCouplets(device,encoder, attn_decoder,batch_in_out_pairs,vocab_size, max_length, hidden_size, encoder_optimizer, decoder_optimizer,int2char_dict,indices_train,indices_valid,ep,longueur)
		# Save models as pt files
		file_name_enc="./models/II_enc_epoch_"+str(ep)+".pt"
		file_name_dec="./models/II_dec_epoch_"+str(ep)+".pt"
		print("\tSaving to ",file_name_enc)
		torch.save(encoder, file_name_enc)
		print("\tSaving to ",file_name_dec)
		torch.save(attn_decoder, file_name_dec)
		
		print("\tTrain Loss: ",train_loss," |  Train PPL: ",math.exp(train_loss))
		print("\tValid Loss: ",valid_loss," |  Valid PPL: ",math.exp(valid_loss))
		lossesPLL_train.append(math.exp(train_loss))
		lossesPLL_valid.append(math.exp(valid_loss))

		end_time = time.time()

		# Display time used
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		print("\tTime: ", epoch_mins ,"m ", epoch_secs, "s")

		print("NOW min lossesPLL_train at epoch:",np.argmin(lossesPLL_train), "value: ",lossesPLL_train[np.argmin(lossesPLL_train)])
		print("NOW min lossesPLL_valid at epoch:",np.argmin(lossesPLL_valid), "value: ",lossesPLL_valid[np.argmin(lossesPLL_valid)])
		print("\t => Train losses(ascending order): ",np.argsort(lossesPLL_train))


