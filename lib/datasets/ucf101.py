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

class ucf101(stackedimdb):
  def __init__(self, image_set, feature, K):
    self.dname = 'ucf101'
    dfilename = 'ucf101_' + feature + '_' + image_set
    self.K = K
    if K > 1:
      dfilename += 'K%d' % K
    # name, paths
    self._image_set = image_set
    datasetroot = osp.join(cfg.DATA_DIR, 'ucf101')
    self._data_path = datasetroot
    self.gtpath = datasetroot + '/detection/gtfile.py'

    if feature == 'rgb':
      self.imagedir = 'images'
    elif feature == 'opf':
      self.imagedir = 'OF_closest'


    self._classes = ('__background__',
                     'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
                     'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing',
                     'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing',
                     'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping',
                     'VolleyballSpiking', 'WalkingWithDog')

    if image_set == 'train' or image_set == 'trainall':
      vidlistpath = datasetroot + '/detection/OF_vidlist_train1.txt'
    elif image_set == 'val' or image_set == 'valall':
      vidlistpath = datasetroot + '/detection/OF_vidlist_test1.txt'

    stackedimdb.__init__(self, dfilename, vidlistpath, image_set)
