# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
import ipdb
import re
import datasets.track_utils

class stackedimdb(imdb):
  def __init__(self, dfilename, vidlistpath, image_set):
    imdb.__init__(self, dfilename, classes=self._classes)

    assert self.dname == "ucf101" or self.dname == "daly"

    if self.dname == "daly":
      self.onkeyframes = True
    else:
      self.onkeyframes = False

    with open(vidlistpath) as f:
      vcontent = f.readlines()
    self.vidlist = [re.sub(' .*', '', x.strip()) for x in vcontent]

    self.on_all_samples = False
    if image_set == 'valall' or image_set == 'trainall':
      self.on_all_samples = True # detect on all video frames
      self.getShots()

    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

    self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self.competition_mode(False) # set cleanup/use_salt

  def loadgtfile(self):
    with open(self.gtpath) as f:
      return pickle.load(f)

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    gtfile = self.loadgtfile()

    self._image_index = []
    # dicts taking index from self._image_index as input key
    self._image_path = {}
    self._widths = {}
    self._heights = {}
    self._idx_2_gtinstance = {}

    K = self.K
    if K > 1:
      self._stack_index = [] # these indices point to the begining of each stack
      if self.on_all_samples:
         self._all_stack_index = []
      self._vid_stack_index = [] # these indices point to the first stack of each video

    self._vid_index = [] # these indices point to the first image of each video

    image_idx = self._image_index
    image_path = self._image_path
    widths = self._widths
    heights = self._heights
    
    idx = 0
    unique_gt_id = 0 # id use to recognize linked GT along time (same GT in a given image stack)
    count_good = 0
    count_bad = 0
    iii = 0
    DEBUG = False
    for vid in self.vidlist: # for all videos from the split
      if iii > 5 and DEBUG:
         break
      iii += 1
      if K > 1:
         # a new video starts at this stack index
         self._vid_stack_index.append(len(self._all_stack_index))
      self._vid_index.append(idx)

      vid_gt = gtfile[vid]
      if 'WH_size' in vid_gt:
        width = vid_gt['WH_size'][0]
        height = vid_gt['WH_size'][1]
      else:
        # default UCF101 size
        if self.dname == "daly":
           raise AssertionError
        width = 320
        height = 240

      vlen = vid_gt['length']

      # get all instance frame spanning
      inst_span = []
      for inst in vid_gt['gts']: # for all video GT instances
        if self.onkeyframes:
           tbounds = []
           for key in inst['keyframes']['keylist']:
              fn = key['frame_num']
              tbound = (fn, fn)
              if K > 1: # get the tracked boxes around the GT instead of the GT annotation
                  # get the GT correspondance in the tracked shot
                  _, gt_i, gbox = datasets.track_utils.framebounds2idx(inst, tbound)
                  max_pix_diff = np.abs(key['boxes'] - gbox).max()
                  max_pix_diff /= float(width*height)/(320*240) # normalize compared to the standard image size
                  if max_pix_diff > 20:
                     #print(datasets.track_utils.framebounds2idx(inst, (fn -3, fn+3)))
                     #print(key['boxes'])
                     #print(max_pix_diff)
                     count_bad += 1
                  else:
                     count_good += 1

                     _idx = []
                     f_bf_gt = gt_i
                     f_af_gt = inst['N_frames'] - gt_i - 1
                     need_bf = int(K/2 - 1 if K % 2 == 0 else K/2)
                     need_af = int(K/2)

                     diff_bf = f_bf_gt - need_bf
                     diff_af = f_af_gt - need_af

                     if diff_bf >= 0 and diff_af >= 0:
                        # just add tracked frames around the GT
                        tbound = (fn - need_bf, fn + need_af)

                     elif diff_bf + diff_af >= 0:
                        # we have enough frames close to the GT
                        if diff_bf < 0:
                           t1 = fn - f_bf_gt # take all frames before
                           t2 = t1 + K - 1 # complete

                        else: # diff_af < 0
                           t2 = fn + f_af_gt # take all frames after
                           t1 = t2 - K + 1 # complete

                        tbound = (t1, t2)

                     # OTHERWISE: tbound remains unchanged (keyframe only), it will not be consider later
              tbounds.append( (tbound, unique_gt_id) )
              unique_gt_id += 1
        else:
           tbound = inst['tbound']
           tbounds = [ (tbound, unique_gt_id) ]
           unique_gt_id += 1
        inst_span.append( tbounds )

      last_stack = {}
      next_shot_end = -1
      shot_id = -1
      for f in range(vlen):
        fn = f + 1 # frame number
        if self.on_all_samples and K > 1 and fn > next_shot_end:
         shot_id += 1
         next_shot_end = self.vid_shot_ends[vid][shot_id]
         assert fn <= next_shot_end

        add_inst = []
        new_stack = False
        _append = False
        for inst_idx, tbounds in enumerate(inst_span): # for each gt instance
         # Note that if not self.onkeyframes, the following loop has a single element: the gt instance
         for key_idx, t_uniq in enumerate(tbounds):
          # for each "instance" or "keyframe of the gt instance"
          tbound, unique_gt_id = t_uniq
          if (fn >= tbound[0]) and (fn <= tbound[1]): # if the frame is inside the gt bound, add the gt instance
            if K > 1: # check the K-1 next GT
               if fn + K - 1 <= tbound[1]: # we can start a new stack from here
                 new_stack = True # the K - 1 next indices will be loaded for this stack
                 last_stack[inst_idx] = 1 # re-init stack count to 1 for this instance
                 _append = True # append this gt
               else: # do not create new stack
                 if inst_idx in last_stack:
                   # if we are stacking this instance
                   _append = True # append the gt
                   last_stack[inst_idx] += 1
                   if last_stack[inst_idx] == K:
                     del last_stack[inst_idx] # stacking is over

            else:
              _append = True # if K == 1, append the frame is inside the gt bound

            if _append:
              if self.onkeyframes:
                 add_inst.append( (inst_idx, unique_gt_id, key_idx) )
              else:
                 add_inst.append( (inst_idx, unique_gt_id) )

        if add_inst or self.on_all_samples: # if the frame contains any gt or if we use all frames
          image_idx.append(idx)
          if new_stack:
            # this position in image_idx is a new stack
            self._stack_index.append(len(image_idx)-1)

          if K > 1 and self.on_all_samples and fn + K - 1 <= next_shot_end:
            # we can start a test stack from here
            self._all_stack_index.append(len(image_idx)-1)

          self._idx_2_gtinstance[idx] = (vid, add_inst, fn)
          widths[idx] = width
          heights[idx] = height
          # get the image path (the full path with be added in image_path_from_index)
          image_path[idx] = '%s/image-%05d' % (vid_gt['videoname'], fn)
          idx += 1

      assert len(last_stack) == 0, 'any started stack has to be finished'

    if self.onkeyframes and K > 1:
       print('We kept %d GT and filtered %d' % (count_good, count_bad) )

  def idx2gtinstance(self, idx, gtfile):
    vid, inst_indices, fn = self._idx_2_gtinstance[idx]

    gtboxes = []
    classes = []
    gtids = [] # unique GT ids given the unique_gt_id of the video
    for ii in inst_indices:
      if self.onkeyframes:
         i, unique_gt_id, i_key = ii
      else:
         i, unique_gt_id = ii

      inst = gtfile[vid]['gts'][i]

      if self.onkeyframes and self.K <= 1:
         keyframe = inst['keyframes']['keylist'][i_key]
         assert keyframe['boxes'].ndim == 1, "we expect only one box"
         box = keyframe['boxes']

      else:
         f_start = inst['tbound'][0] # gt starting frame
         box_id = fn - f_start
         assert box_id >= 0
         box = inst['boxes'][box_id, :]

      gtboxes.append(box)
      classes.append(inst['label']) # in annotation: starts at one with bkg is last class, so class idx is correct
      gtids.append(unique_gt_id)

    return gtboxes, classes, gtids

  def _get_widths(self):
    return [ self._widths[idx] for idx in self._image_index ]

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = '%s/%s/%s.jpg' % (self._data_path, self.imagedir, self._image_path[index])
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gtfile = self.loadgtfile()
    gt_roidb = [self._load_stacked_annotation(index, gtfile)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_stacked_annotation(self, index, gtfile):
    """
    Loads bounding-box instance annotations.
    """
    gtboxes, classes, gtids = self.idx2gtinstance(index, gtfile)
    width = self._widths[index]
    height = self._heights[index]

    num_objs = len(gtboxes)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32) # 1 at the gt class, 0 otherwise
    seg_areas = np.zeros((num_objs), dtype=np.float32) # box area here
    gt_ids = np.array(gtids, dtype=int)

    for i, ibox in enumerate(gtboxes):
      bbox = ibox - 1 # start at 0
      x1 = bbox[0]
      y1 = bbox[1]
      x2 = bbox[2]
      y2 = bbox[3]

      boxes[i, :] = bbox
      cls = classes[i]
      gt_classes[i] = cls
      overlaps[i, cls] = 1
      seg_areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)
    ds_utils.validate_boxes(boxes, width=width, height=height)

    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'gt_ids': gt_ids,
            'flipped': False,
            'seg_areas': seg_areas}

  def append_flipped_images(self):
    # redefine this function in ordrer to add more fields
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'gt_ids': self.roidb[i]['gt_ids'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2
    if self.K > 1:
      # make the 2nd part of _stack_index point on
      # flipped frames (2nd part of roidb)
      slen = len(self._image_index) / 2
      sdup = [ i+slen for i in self._stack_index ]
      self._stack_index = self._stack_index + sdup

  def evaluate_detections(self, all_boxes, output_dir):
    raise NotImplementedError
    res_file = osp.join(output_dir, ('detections_' +
                                     self._image_set +
                                     self._year +
                                     '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    self._write_coco_results_file(all_boxes, res_file)
    # Only do evaluation on non-test sets
    if self._image_set.find('test') == -1:
      self._do_detection_eval(res_file, output_dir)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True
