from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import sys
sys.path.append("nlg-eval")
from nlgeval import compute_metrics

def language_eval(dataset, preds, model_id, split):
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    
    label = {}
    imgs = json.load(open(dataset, 'r'))
    for img in imgs:
        if img["split"] == split:
            label[img["dataid"]] = img
    
    pred_ids = []
    with open("eval_results/preds.txt", "w") as text_file:
        for pred in preds:
            pred_id = pred["image_id"]
            pred_txt = pred["caption"]
            pred_ids.append(pred_id)
            text_file.write(pred_txt + "\n")

    with open("eval_results/refer.txt", "w") as text_file:
        for pred_id in pred_ids:
            raw_sent = label[pred_id]["sentences"][0]["raw"].lower().encode('utf-8')
            text_file.write(raw_sent + "\n")
    
    metrics_dict = compute_metrics(hypothesis='eval_results/preds.txt',
                               references=['eval_results/refer.txt'])

    return metrics_dict

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    beam_size = eval_kwargs.get('beam_size', 1)
    dataset = eval_kwargs["ana_json"]

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    have_dumped = False
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        if data.get('labels', None) is not None:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks = tmp

            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image
        seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)
        
        #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        dum_count = 100
        dum_i = dum_count
        if eval_kwargs.get('dump_images', 0) == 1 and not have_dumped:
            os.system("rm vis/imgs/* -f") 

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                if dum_i > 0 and not have_dumped:
                    # dump the raw image to vis/ folder
                    cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                    print(cmd)
                    os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

            dum_i -= 1

        if eval_kwargs.get('dump_images', 0) == 1 and not have_dumped:
            json.dump(predictions[0:(dum_count - 1)], open('vis/vis.json', 'w'))
            have_dumped = True

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
