# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.stackedimdb import stackedimdb
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

class daly(stackedimdb):
  def __init__(self, image_set, feature, K):
    self.dname = 'daly'
    dfilename = 'daly_' + feature + '_' + image_set
    self.K = K
    if K > 1:
      dfilename += 'K%d' % K
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


    self.shotpath = '/sequoia/data2/gcheron/DALY/daly_shotdet/mat_shots/'

    if image_set == 'train' or image_set == 'trainall':
      vidlistpath = datasetroot + '/OF_vidlist_train1.txt'
    elif image_set == 'val' or image_set == 'valall':
      vidlistpath = datasetroot + '/OF_vidlist_test1.txt'

    stackedimdb.__init__(self, dfilename, vidlistpath, image_set)

  def getShots(self):
      self.vid_shot_ends = {}
      gtfile = self.loadgtfile()
      for vid in self.vidlist:
         shots = sio.loadmat(self.shotpath + vid + '.mp4.mat')['shots']
         _cmp =  gtfile[vid]['length']
         assert shots[0, 0] == 1
         if not shots[-1, -1] == _cmp:
            print('%s: clamp last frame %d --> %d' % (vid, shots[-1, -1], _cmp))
            shots[-1, -1] = _cmp

         assert shots.shape[1] == 2 and shots.ndim == 2
         self.vid_shot_ends[vid] = [ int(shots[i, 1]) for i in range(len(shots)) ]
