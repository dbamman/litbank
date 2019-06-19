import os,sys,argparse
import pytorch_pretrained_bert
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import sequence_layered_reader, sequence_eval, sequence_reader
import numpy as np
from tagger import Tagger
import flat_reader, layered_reader

batch_size=32
dropout_rate=0.25
bert_dim=768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('')
print("********************************************")
print("Running on: {}".format(device))
print("********************************************")
print('')

def predict(predictFile, outFile, model, ignoreEvents, ignoreEntities):

	testSentences, origTestSentences = sequence_layered_reader.read_booknlp(predictFile, model)

	with open(outFile, "w", encoding="utf-8") as out:

		# FLAT

		if not ignoreEvents:

			batched_sents_flat, batched_orig_token_lens_flat, batched_data_flat, batched_mask_flat, batched_transforms_flat, ordering_flat=flat_reader.get_batches(model, testSentences, batch_size, training=False)
			flat_preds_in_order=model.tagFlat(batched_sents_flat, batched_data_flat, batched_mask_flat, batched_transforms_flat, batched_orig_token_lens_flat, ordering_flat)

			for s_idx, preds in enumerate(flat_preds_in_order):
				for w_idx, val in enumerate(preds):
					(word, label)=val
					# account for [CLS]
					o_tid, o_word=origTestSentences[s_idx][w_idx+1]
					
					assert (word == o_word)
					
					if label == 1:
						out.write("%s\t%s\t%s\t%s\n" % (o_tid, o_tid, "EVENT", word))


		# LAYERED

		if not ignoreEntities:

			batched_sents, batched_data, batched_mask, batched_transforms, batched_orig_token_lens, ordering = layered_reader.get_batches(model, testSentences, batch_size, tagset, training=False)
			preds_in_order=model.tag(batched_sents, batched_data, batched_mask, batched_transforms, batched_orig_token_lens, ordering)

			for idx, preds in enumerate(preds_in_order):
				for _, label, start, end in preds:
					# account for [CLS]
					start+=1
					end+=1
					start_token=origTestSentences[idx][start][0]
					end_token=origTestSentences[idx][end][0]
					phrase=' '.join([x[0] for x in testSentences[idx][start:end]])
					phraseEndToken=int(end_token)-1
					if phraseEndToken == -2:
						phraseEndToken=start_token
					out.write("%s\t%s\t%s\t%s\n" % (start_token, phraseEndToken, label, phrase))



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', help='{train,test,predict,predictBatch}', required=True)
	
	parser.add_argument('--batch_prediction_file', help='Filename containing input paths to tag, paired with output paths to write to', required=False)
	
	parser.add_argument('-i','--input_prediction_file', help='Filename to tag', required=False)
	parser.add_argument('-o','--output_prediction_file', help='Filename to write tagged text to', required=False)

	parser.add_argument('--trainFolder_flat', help='Folder containing training data (flat)', required=False)
	parser.add_argument('--testFolder_flat', help='Folder containing test data (flat)', required=False)
	parser.add_argument('--devFolder_flat', help='Folder containing dev data (flat)', required=False)

	parser.add_argument('--trainFolder_layered', help='Folder containing training data (layered)', required=False)
	parser.add_argument('--testFolder_layered', help='Folder containing test data (layered)', required=False)
	parser.add_argument('--devFolder_layered', help='Folder containing dev data (layered)', required=False)

	parser.add_argument('--tagFile_flat', help='File mapping tags to tag ids (flat)', default="files/event.tagset", required=False)
	parser.add_argument('--tagFile_layered', help='File mapping tags to tag ids (layered)', default="files/entity.tagset", required=False)

	parser.add_argument('--modelFile', help='File to write model to/read from', default="files/event.entity.model", required=False)
	parser.add_argument('--flat_metric', help='{accuracy,fscore,span_fscore}', required=False)

	parser.add_argument('--ignoreEvents', action="store_true", default=False, help='tag events', required=False)
	parser.add_argument('--ignoreEntities', action="store_true", default=False, help='tag entities', required=False)

	args = vars(parser.parse_args())

	print(args)

	mode=args["mode"]
	
	tagset_flat=sequence_layered_reader.read_tagset(args["tagFile_flat"])
	tagset=sequence_layered_reader.read_tagset(args["tagFile_layered"])

	model_file=args["modelFile"]

	cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(0))

	model = Tagger.from_pretrained('bert-base-cased',
			  cache_dir=cache_dir, freeze_bert=True, tagset_flat=tagset_flat, tagset=tagset, device=device)

	model.to(device)

	if mode == "train":

		train_folder_flat=args["trainFolder_flat"]
		dev_folder_flat=args["devFolder_flat"]
	
		train_folder_layered=args["trainFolder_layered"]
		dev_folder_layered=args["devFolder_layered"]

		flat_metric=None
		if args["flat_metric"].lower() == "fscore":
			flat_metric=sequence_eval.check_f1_two_lists
		elif args["flat_metric"].lower() == "accuracy":
			flat_metric=sequence_eval.get_accuracy
		elif args["flat_metric"].lower() == "span_fscore":
			flat_metric=sequence_eval.check_span_f1_two_lists

		## Flat

		trainSentences = sequence_reader.prepare_annotations_from_folder(train_folder_flat, tagset_flat)
		devSentences = sequence_reader.prepare_annotations_from_folder(dev_folder_flat, tagset_flat)
		
		batched_sents_flat, batched_orig_token_lens_flat, batched_data_flat, batched_mask_flat, batched_labels_flat, batched_transforms_flat, ordering_flat=flat_reader.get_batches(model, trainSentences, batch_size)
		dev_batched_sents_flat, dev_batched_orig_token_lens_flat, dev_batched_data_flat, dev_batched_mask_flat, dev_batched_labels_flat, dev_batched_transforms_flat, dev_ordering_flat=flat_reader.get_batches(model, devSentences, batch_size)


		# NESTED NER

		trainSentences_nested = sequence_layered_reader.prepare_annotations_from_folder(train_folder_layered, tagset)
		devSentences_nested = sequence_layered_reader.prepare_annotations_from_folder(dev_folder_layered, tagset)
		
		batched_sents, batched_data, batched_mask, batched_labels, batched_transforms, ordering, batched_layered_labels1, batched_layered_labels2, batched_layered_labels3, batched_layered_labels4, batched_layered_labels5, batched_index1, batched_index2, batched_index3, batched_newlabel1, batched_newlabel2, batched_newlabel3, lens=layered_reader.get_batches(model, trainSentences_nested, batch_size, tagset)
		dev_batched_sents, dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_ordering, dev_batched_layered_labels1, dev_batched_layered_labels2, dev_batched_layered_labels3, dev_batched_layered_labels4, dev_batched_layered_labels5, dev_batched_index1, dev_batched_index2, dev_batched_index3, dev_batched_newlabel1, dev_batched_newlabel2, dev_batched_newlabel3, dev_lens=layered_reader.get_batches(model, devSentences_nested, batch_size, tagset)

		optimizer = optim.Adam(model.parameters())		
		
		maxScore=0
		best_idx=0
		patience=10

		for epoch in range(100):
			model.train()

			for b in range(len(batched_data_flat)):
				if b % 10 == 0:
					print(b)
					sys.stdout.flush()
				
				loss_flat = model.forwardFlatSequence(batched_data_flat[b], token_type_ids=None, attention_mask=batched_mask_flat[b], transforms=batched_transforms_flat[b], labels=batched_labels_flat[b])

				loss_flat.backward()
				optimizer.step()
				model.zero_grad()

			print("***FLAT EVAL***")

			score=model.evaluateFlat(dev_batched_data_flat, dev_batched_mask_flat, dev_batched_labels_flat, dev_batched_transforms_flat, flat_metric, tagset_flat)

			sys.stdout.flush()

			if score > maxScore:
				torch.save(model.state_dict(), model_file)
				print("%s is better than %s, saving..." % (score, maxScore))
				maxScore=score
				best_idx=epoch

			print(epoch-best_idx, patience, epoch, best_idx)

			if epoch-best_idx > patience:
				print ("Stopping flat training at epoch %s" % epoch)
				break


		maxScore=0
		best_idx=0
		patience/=2

		# load best event model so far (not current state)
		model = Tagger.from_pretrained('bert-base-cased',
			  cache_dir=cache_dir, freeze_bert=True, tagset_flat=tagset_flat, tagset=tagset, device=device)

		model.to(device)
		model.load_state_dict(torch.load(model_file, map_location=device))
		optimizer = optim.Adam(model.parameters())	

		for epoch in range(100):

			model.train()

			for b in range(len(batched_data)):
				if b % 10 == 0:
					print(b)
					sys.stdout.flush()
				
				loss_layered = model.forward(batched_data[b], batched_index1[b], batched_index2[b], attention_mask=batched_mask[b], transforms=batched_transforms[b], labels=[batched_newlabel1[b], batched_newlabel2[b], batched_newlabel3[b]], lens=[lens[0][b], lens[1][b], lens[2][b] ])

				loss_layered.backward()
				optimizer.step()
				model.zero_grad()

			print("***LAYERED EVAL***")

			score=model.evaluate(dev_batched_sents, dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_batched_layered_labels1, dev_batched_layered_labels2, dev_batched_layered_labels3, dev_batched_layered_labels4, dev_lens)

			sys.stdout.flush()

			if score > maxScore:
				torch.save(model.state_dict(), model_file)
				print("%s is better than %s, saving..." % (score, maxScore))
				maxScore=score
				best_idx=epoch

			print(epoch-best_idx, patience, epoch, best_idx)
			if epoch-best_idx > patience:
				print ("Stopping layered training at epoch %s" % epoch)
				break

	elif mode == "test":

		dev_folder_flat=args["testFolder_flat"]
		dev_folder_layered=args["testFolder_layered"]

		flat_metric=None
		if args["flat_metric"].lower() == "fscore":
			flat_metric=sequence_eval.check_f1_two_lists
		elif args["flat_metric"].lower() == "accuracy":
			flat_metric=sequence_eval.get_accuracy
		elif args["flat_metric"].lower() == "span_fscore":
			flat_metric=sequence_eval.check_span_f1_two_lists


		## FLAT

		devSentences = sequence_reader.prepare_annotations_from_folder(dev_folder_flat, tagset_flat)
		dev_batched_sents_flat, dev_batched_orig_token_lens_flat, dev_batched_data_flat, dev_batched_mask_flat, dev_batched_labels_flat, dev_batched_transforms_flat, dev_ordering_flat=flat_reader.get_batches(model, devSentences, batch_size)


		# NESTED NER

		devSentences_nested = sequence_layered_reader.prepare_annotations_from_folder(dev_folder_layered, tagset)
		dev_batched_sents, dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_ordering, dev_batched_layered_labels1, dev_batched_layered_labels2, dev_batched_layered_labels3, dev_batched_layered_labels4, dev_batched_layered_labels5, dev_batched_index1, dev_batched_index2, dev_batched_index3, dev_batched_newlabel1, dev_batched_newlabel2, dev_batched_newlabel3, dev_lens=layered_reader.get_batches(model, devSentences_nested, batch_size, tagset)

		model.load_state_dict(torch.load(model_file, map_location=device))

		print("***FLAT EVAL***")

		score=model.evaluateFlat(dev_batched_data_flat, dev_batched_mask_flat, dev_batched_labels_flat, dev_batched_transforms_flat, flat_metric, tagset_flat)
		
		print("***LAYERED EVAL***")

		score=model.evaluate(dev_batched_sents, dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_batched_layered_labels1, dev_batched_layered_labels2, dev_batched_layered_labels3, dev_batched_layered_labels4, dev_lens)



	elif mode == "predict":

		predictFile=args["input_prediction_file"]
		outFile=args["output_prediction_file"]

		model.load_state_dict(torch.load(model_file, map_location=device))

		predict(predictFile, outFile, model, bool(args["ignoreEvents"]), bool(args["ignoreEntities"]))


	elif mode == "predictBatch":

		pathFile=args["batch_prediction_file"]

		model.load_state_dict(torch.load(model_file))

		inpaths, outpaths=sequence_layered_reader.read_filenames(pathFile)
		for infile, outfile in zip(inpaths, outpaths):
			print("Tagging %s" % infile)
			sys.stdout.flush()
			predict(infile, outfile, model, bool(args["ignoreEvents"]), bool(args["ignoreEntities"]))




