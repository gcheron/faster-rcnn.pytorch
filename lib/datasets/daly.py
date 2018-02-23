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

class daly(imdb):
  def __init__(self, image_set, feature):
    imdb.__init__(self, 'daly_' + feature + '_' + image_set)
    # name, paths
    self._image_set = image_set
    datasetroot = osp.join(cfg.DATA_DIR, 'daly')
    self._data_path = datasetroot
    self.gtpath = datasetroot + '/gtfile.pkl'

    if feature == 'rgb':
      self.imagedir = 'images'
    elif feature == 'opf':
      self.imagedir = 'OF_closest'


    self._classes = ('__background__',
                     'ApplyingMakeUpOnLips', 'BrushingTeeth', 'CleaningFloor', 'CleaningWindows', 'Drinking',
                     'FoldingTextile', 'Ironing', 'Phoning', 'PlayingHarmonica', 'TakingPhotosOrVideos')

    gtfile = self.loadgtfile()

    if image_set == 'train' or image_set == 'trainall':
      vidlistpath = datasetroot + '/OF_vidlist_train1.txt'
    elif image_set == 'val' or image_set == 'valall':
      vidlistpath = datasetroot + '/OF_vidlist_test1.txt'
    with open(vidlistpath) as f:
      vcontent = f.readlines()
    self.vidlist = [re.sub(' .*', '', x.strip()) for x in vcontent]

    self.on_all_samples = False
    if image_set == 'valall' or image_set == 'trainall':
      self.on_all_samples = True # detect on all video frames

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

    image_idx = self._image_index
    image_path = self._image_path
    widths = self._widths
    heights = self._heights
    
    idx = 0
    for vid in self.vidlist: # for all videos from the split
      vid_gt = gtfile[vid]
      if 'WH_size' in vid_gt:
        width = vid_gt['WH_size'][0]
        height = vid_gt['WH_size'][1]
      else:
        # default UCF101 size
        raise AssertionError
        width = 320
        height = 240

      vlen = vid_gt['length']

      # get all instance frame spanning
      inst_span = []
      for inst in vid_gt['gts']: # for all video GT instances
        #tbound = inst['tbound']
        #inst_span.append(tbound)
        tbounds = []
        for key in inst['keyframes']['keylist']:
           fn = key['frame_num']
           tbounds.append( (fn, fn) )
        inst_span.append(tbounds)
      
      for f in range(vlen):
        fn = f + 1 # frame number
        add_inst = []
        #for inst_idx, tbound in enumerate(inst_span):
        for inst_idx, tbounds in enumerate(inst_span): # for each gt instance
           for key_idx, tbound in enumerate(tbounds): # for each keyframe of the gt instance
             if (fn >= tbound[0]) and (fn <= tbound[1]): # if the frame is inside the gt bound, add the gt instance
               #add_inst.append(inst_idx)
               add_inst.append( (inst_idx, key_idx) )

        if add_inst or self.on_all_samples: # if the frame contains any gt or if we use all frames
          image_idx.append(idx)
          self._idx_2_gtinstance[idx] = (vid, add_inst, fn)
          widths[idx] = width
          heights[idx] = height
          # get the image path (the full path with be added in image_path_from_index)
          image_path[idx] = '%s/image-%05d' % (vid_gt['videoname'], fn)
          idx += 1

  def idx2gtinstance(self, idx, gtfile):
    vid, inst_indices, fn = self._idx_2_gtinstance[idx]

    gtboxes = []
    classes = []
    #for i in inst_indices:
    for i, i_key in inst_indices: # instance and its key index
      inst = gtfile[vid]['gts'][i]

      keyframe = inst['keyframes']['keylist'][i_key]

      #f_start = inst['tbound'][0] # gt starting frame
      #box_id = fn - f_start
      #assert box_id >= 0
      #gtboxes.append(inst['boxes'][box_id, :])
      assert keyframe['boxes'].ndim == 1, "we expect only one box"
      gtboxes.append(keyframe['boxes'])
      classes.append(inst['label']) # in annotation: starts at one with bkg is last class, so class idx is correct

    return gtboxes, classes

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
    gt_roidb = [self._load_daly_annotation(index, gtfile)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_daly_annotation(self, index, gtfile):
    """
    Loads DALY bounding-box instance annotations.
    """
    gtboxes, classes = self.idx2gtinstance(index, gtfile)
    width = self._widths[index]
    height = self._heights[index]

    num_objs = len(gtboxes)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32) # 1 at the gt class, 0 otherwise
    seg_areas = np.zeros((num_objs), dtype=np.float32) # box area here

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
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

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
