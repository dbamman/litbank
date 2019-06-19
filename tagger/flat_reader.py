import numpy as np
import torch

def get_batches(model, sentences, max_batch, training=True):
		"""
		Partitions a list of sentences (each a list containing [word, label]) into a set of batches
		Returns:
		
		-- batched_sents: original tokens in sentences
		
		-- batched_orig_token_lens: length of original tokens in sentences
		
		-- batched_data: token ids of sentences. [[101 37 42 102], [101 7 102 0]]
		
		-- batched_mask: Binary flag for real tokens (1) and padded tokens (0) [[1 1 1 1], [1 1 1 0]]
		
		-- batched_transforms: BERT word piece tokenization splits words into pieces; this matrix specifies how
		to combine those pieces back into the original tokens (by averaging their representations) using matrix operations.
		If the original sentence is 3 words that have been tokenized into 4 word piece tokens [101 37 42 102] 
		(where 37 42 are the pieces of one original word), the transformation matrix is 4 x 3 (zero padded to 4 x 4), 
		resulting in the original sequence length of 3. [[1 0 0 0], [0 0.5 0.5 0], [0 0 0 1]]. 

		-- batched_labels (when training=True): Labels for each sentence, one label per original token (prior to 
		word piece tokenization). Loss function ignores padded tokens with -100 label. 

		-- ordering: inverse argsort to recover original ordering of sentences.

		"""
		
		all_sents=[]
		all_orig_token_lens=[]
		maxLen=0
		for sentence in sentences:
			
			ts=[x[0] for x in sentence]
			all_sents.append(ts)
			
			all_orig_token_lens.append(len(sentence))
			length=0
			for word in sentence:
				toks=model.tokenizer.tokenize(word[0])
				length+=len(toks)

			if length> maxLen:
				maxLen=length

		all_data=[]
		all_masks=[]
		all_labels=[]
		all_transforms=[]

		for sentence in sentences:
			tok_ids=[]
			input_mask=[]
			labels=[]
			transform=[]

			all_toks=[]
			n=0
			for idx, word in enumerate(sentence):
				toks=model.tokenizer.tokenize(word[0])
				all_toks.append(toks)
				n+=len(toks)

			cur=0
			for idx, word in enumerate(sentence):
				toks=all_toks[idx]
				ind=list(np.zeros(n))
				for j in range(cur,cur+len(toks)):
					ind[j]=1./len(toks)
				cur+=len(toks)
				transform.append(ind)

				tok_ids.extend(model.tokenizer.convert_tokens_to_ids(toks))

				input_mask.extend(np.ones(len(toks)))
				
				if training:
					labels.append(int(word[1]))

			all_data.append(tok_ids)
			all_masks.append(input_mask)
			all_transforms.append(transform)

			if training:
				all_labels.append(labels)

		lengths = np.array([len(l) for l in all_data])
		ordering = np.argsort(lengths)
		ordered_data = [None for i in range(len(all_data))]
		ordered_masks = [None for i in range(len(all_data))]
		ordered_transforms = [None for i in range(len(all_data))]
		orig_token_lens = [None for i in range(len(all_data))]
		orig_sents = [None for i in range(len(all_data))]

		if training:		
			ordered_labels = [None for i in range(len(all_data))]

		for i, ind in enumerate(ordering):
			ordered_data[i] = all_data[ind]
			ordered_masks[i] = all_masks[ind]
			ordered_transforms[i] = all_transforms[ind]
			orig_token_lens[i]=all_orig_token_lens[ind]
			orig_sents[i]=all_sents[ind]

			if training:
				ordered_labels[i] = all_labels[ind]

		batched_data=[]
		batched_mask=[]
		batched_transforms=[]
		batched_orig_token_lens=[]
		batched_sents=[]

		if training:
			batched_labels=[]

		i=0

		current_batch=max_batch

		while i < len(ordered_data):

			batch_data=ordered_data[i:i+current_batch]
			batch_mask=ordered_masks[i:i+current_batch]
			batch_transforms=ordered_transforms[i:i+current_batch]
			batch_orig_lens=orig_token_lens[i:i+current_batch]
			batch_sents=orig_sents[i:i+current_batch]

			max_label_length = max([l for l in batch_orig_lens])
			max_len = max([len(sent) for sent in batch_data])


			if training:
				batch_labels=ordered_labels[i:i+current_batch]

				max_label = max([len(label) for label in batch_labels])

			for j in range(len(batch_data)):
				
				blen=len(batch_data[j])

				if training:
					blab=len(batch_labels[j])

				for k in range(blen, max_len):
					batch_data[j].append(0)
					batch_mask[j].append(0)
					for z in range(len(batch_transforms[j])):
						batch_transforms[j][z].append(0)

				if training:
					for k in range(blab, max_label_length):
						batch_labels[j].append(-100)

				for k in range(len(batch_transforms[j]), max_label_length):
					batch_transforms[j].append(np.zeros(max_len))

			batched_data.append(torch.LongTensor(batch_data))
			batched_mask.append(torch.FloatTensor(batch_mask))
			batched_transforms.append(torch.FloatTensor(batch_transforms))
			batched_orig_token_lens.append(torch.LongTensor(batch_orig_lens))
			batched_sents.append(batch_sents)

			if training:
				batched_labels.append(torch.LongTensor(batch_labels))
			
			i+=current_batch

			if max_len > 100:
				current_batch=12
			if max_len > 200:
				current_batch=6


		if training:
			return batched_sents, batched_orig_token_lens, batched_data, batched_mask, batched_labels, batched_transforms, ordering

		return batched_sents, batched_orig_token_lens, batched_data, batched_mask, batched_transforms, ordering

