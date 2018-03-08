import numpy as np
import pickle
import ipdb
import re
from tube_utils import iou2d
import os
import glob

class tube_builder():
   def __init__(self, detpath, resdir, nclasses, K, per_video_dets=True, hasShots=True):
      self.resdir = resdir
      self.nclasses = nclasses
      self.K = K
      self.nms = 0.3 # nms on tubelets
      self.top_k = 10 # max nb of tubelets to keep at each it
      self.merge_iou = 0.2 # min iou to merge a tubelet with tube
      self.offset_end = 5 # frame number without merging tubelet after which a tube ends
      self.min_tube_score = 0.01 # min score to keep the final tube
      self.min_tube_length = 15 # min length to keep the final tube
      self.per_video_dets = per_video_dets # we have one detection file per video
      self.hasShots = hasShots

      if self.per_video_dets:
         self.detdir = detpath
         self.vidlist = glob.glob(self.detdir + '/*')
         print 'found %d videos' % len(self.vidlist)
      else:
         self.detfile = detpath
         with open(self.detfile) as f:
            self.detections = pickle.load(f)
         assert len(self.detections) == self.nclasses + 2 # bkg + detpath
      if self.hasShots:
         assert self.per_video_dets, 'please provide per video detections when several shots'

      if not os.path.exists(resdir):
         os.makedirs(resdir)

   def nms_tubelets(self, dets):
      """Compute the NMS for a set of scored tubelets
      scored tubelets are numpy array with 4K+1 columns, last one being the score
      return the indices of the tubelets to keep
      """
      overlapThresh = self.nms
      top_k = self.top_k
   
      # If there are no detections, return an empty list
      if len(dets) == 0: return np.empty((0,), dtype=np.int32)
      if top_k is None: top_k = len(dets)
   
      pick = []
   
      K = (dets.shape[1] - 1) / 4
   
      # Coordinates of bounding boxes
      x1 = [dets[:, 4*k] for k in xrange(K)]
      y1 = [dets[:, 4*k + 1] for k in xrange(K)]
      x2 = [dets[:, 4*k + 2] for k in xrange(K)]
      y2 = [dets[:, 4*k + 3] for k in xrange(K)]
   
      # Compute the area of the bounding boxes and sort the bounding
      # boxes by score
      # area = (x2 - x1 + 1) * (y2 - y1 + 1)
      area = [(x2[k] - x1[k] + 1) * (y2[k] - y1[k] + 1) for k in xrange(K)]
      I = np.argsort(dets[:,-1])
      indices = np.empty(top_k, dtype=np.int32)
      counter = 0
   
      while I.size > 0:
          i = I[-1]
          indices[counter] = i
          counter += 1
   
          # Compute overlap
          xx1 = [np.maximum(x1[k][i], x1[k][I[:-1]]) for k in xrange(K)]
          yy1 = [np.maximum(y1[k][i], y1[k][I[:-1]]) for k in xrange(K)]
          xx2 = [np.minimum(x2[k][i], x2[k][I[:-1]]) for k in xrange(K)]
          yy2 = [np.minimum(y2[k][i], y2[k][I[:-1]]) for k in xrange(K)]
   
          w = [np.maximum(0, xx2[k] - xx1[k] + 1) for k in xrange(K)]
          h = [np.maximum(0, yy2[k] - yy1[k] + 1) for k in xrange(K)]
   
          inter_area = [w[k] * h[k] for k in xrange(K)]
          ious = sum([inter_area[k] / (area[k][I[:-1]] + area[k][i] - inter_area[k]) for k in xrange(K)])
   
          I = I[np.where(ious <= overlapThresh * K)[0]]
   
          if counter == top_k: break
   
      return indices[:counter]

   def tubescore(self, tube):
      return np.mean(np.array( [ tube[i][1][-1] for i in xrange(len(tube)) ] ))

   def fillmissing(self, tube, idx):
      assert np.logical_not(idx).any(), 'at least one box has to be filled!'

      prev_sbox = None
      fill_idx = []
      for f, fill in enumerate(idx):
         if fill:
            fill_idx.append(f) 
         else:
            # we found a valid box
            next_sbox = tube[f][1:6]
            if fill_idx:
               # there are boxes to interpolate
               numb = len(fill_idx)
               for b in range(numb):
                  i_b = fill_idx[b]
                  if prev_sbox is None:
                     # we did not find boxes before
                     assert fill_idx[0] == 1
                     # just copy the next box/score
                     tube[i_b][1:6] = next_sbox

                  else:
                     # linearly fill all boxes
                     w1 = float(numb - b) / ( numb + 1)
                     w2 = 1 - w1
                     tube[i_b][1:6] = w1 * prev_sbox + w2 * next_sbox

            fill_idx = []
            prev_sbox = next_sbox

      if fill_idx: # prev boxes have to be filled
         # just copy the prev box/score we found
         for b in range(len(fill_idx)):
            tube[ fill_idx[b] ][1:6] = prev_sbox


   def build_tubes(self):
      K = self.K
      tubescore = self.tubescore

      if self.per_video_dets:
         num_dets = len(self.vidlist)
      else:
         num_dets = len(self.detections[0])
      # split detections per video
      prev_vid = ''
      cur_im = -1
      video_dets = {}
      for i_det in xrange(num_dets):
         if self.per_video_dets:
            cur_vid = self.vidlist[i_det]
            cur_vid = re.match('.*/([^/]*).pkl',cur_vid).group(1)
         else:
            cur_vid = self.detections[-1][i_det]
            assert len(cur_vid) == 1
            rres = re.match('.*/([^/]*)/image-([0-9]*)\.*',cur_vid[0])
            cur_vid, cur_im = rres.group(1), int(rres.group(2))

         if self.per_video_dets or cur_vid != prev_vid:
            assert not cur_vid in video_dets
            video_dets[cur_vid] = [[] for _ in xrange(self.nclasses)]
            if not self.per_video_dets:
               assert cur_im == 1
         else:
            assert cur_im == prev_im + 1

         for c in xrange(self.nclasses):
            if self.per_video_dets:
               video_dets[cur_vid][0] = self.vidlist[i_det]
            else:
               cdets = self.detections[c+1][i_det] # skip bkg
               video_dets[cur_vid][c].append(cdets)

         prev_vid = cur_vid
         prev_im = cur_im

      # link tubelets into tubes
      for i_v, vid in enumerate(video_dets):
         if self.per_video_dets:
            with open(video_dets[vid][0]) as f:
               vdets = pickle.load(f)
            assert len(vdets[0]) == self.nclasses + 2 # bkg + detpath

            if self.hasShots:
               # grab image paths to get shots
               paths = [ vdets[f][-1] for f in range(len(vdets)) ]
               vid_shots = []
               for i_f_det, _impath in enumerate(paths):
                  fn = int(re.match('.*/[^/]*/image-([0-9]*)\.*',_impath[0]).group(1))
                  if i_f_det == 0:
                     f_start = fn
                  elif fn != prev_im + 1:
                     assert fn > prev_im
                     # this shot ends at index (i_f_det - 1) and f_start (prev_im) is
                     # the first frame of its first (last) stack
                     vid_shots.append( (i_f_det - 1, f_start, prev_im) )
                     f_start = fn # the new shot starts here

                  prev_im = fn

               vid_shots.append( (i_f_det, f_start, prev_im) ) # get last shot

               # check shots
               prev_end = -K + 1
               p_i_end = -1
               for i_shot, cshot in enumerate(vid_shots):
                  i_f_detend, f_start, f_end = cshot
                  numdets = f_end - f_start + 1 # number of detections in this shot
                  numsince = i_f_detend - p_i_end # number of detections since the previous shot
                  assert numdets == numsince
                  if f_start != prev_end + K:
                     # one shot has been skipped
                     skip_start = prev_end + K
                     skip_end = f_start - 1
                     slen = skip_end - skip_start + 1
                     print '%s: shot from frame %d --> %d (length %d) has been skipped' % (
                     vid, skip_start, skip_end, slen)
                     assert slen < K, 'skipped shot length must be 0 < (%d) < K' % slen
                  prev_end = f_end
                  p_i_end = i_f_detend

            # reorder: nclass x dets
            _tmp = []
            for c in xrange(self.nclasses):
               # skip bkg
               _tmp.append( [ vdets[f][c+1] for f in range(len(vdets)) ] )
            vdets = _tmp
         else:
            vdets = video_dets[vid]

         outfile = '%s/%s.pkl' % (self.resdir, vid)
         n_stack_dets = len(vdets[0])

         if i_v % 50 == 0:
            print '%d/%d: save %s' % (i_v + 1, len(video_dets), outfile)
         
         res = {}
         for c in xrange(self.nclasses):
            finished_tubes = []
            cur_tubes = []

            i_shot = -1
            last_of_shot = True
            for i_d in xrange(n_stack_dets):
               if self.hasShots:
                  if last_of_shot:
                     i_shot += 1 # the previous shot ended

                  # get shot info
                  i_last_detshot, fshot_start, fshot_end = vid_shots[i_shot]

                  # the previous shot ended
                  if last_of_shot: # we start a new shot
                     frame = fshot_start
                     last_of_shot = False
                     init_tubes = True
                  else:
                     frame += 1
                     init_tubes = False

                  # check if this shot ends
                  if i_d == i_last_detshot: # this is the last detection of the shot
                     last_of_shot = True

               else:
                  frame = i_d + 1
               # get tubelets and NMS
               tubelets = vdets[c][i_d] # get K boxes and score
               idx = self.nms_tubelets(tubelets)
               tubelets = tubelets[idx, :]

               if i_d == 0 or (self.hasShots and init_tubes):
                  assert len(cur_tubes) == 0
                  if self.hasShots and last_of_shot:
                     # this is a small shot composed of one stack only, skip it
                     assert fshot_end - fshot_start + 1 == K
                     print '%s: skip one-stack shot of frame %d --> %d' % (vid, fshot_start, fshot_end)
                  else:
                     # start tubes
                     for i in xrange(tubelets.shape[0]):
                        cur_tubes.append( [(frame, tubelets[i, :])] )

                  continue

               # sort tube according to scores
               tubescores = [ tubescore(tube) for tube in cur_tubes ]
               idx = np.argsort(-np.array(tubescores))
               cur_tubes = [cur_tubes[i] for i in idx]

               finished = []
               for i_t, tube in enumerate(cur_tubes): # for each tube
                  # compute ious between tube's last tubelet and tubelets
                  last_frame, last_tubelet = tube[-1]
                  offset = frame - last_frame
                  if offset < K:
                     # there is overlap between current frame and tube's last tubelet
                     nov = K - offset
                     ious = 0
                     for i_o in range(nov):
                        # get overlapping boxes from tube and tubelets
                        tube_box = last_tubelet[4*(i_o+offset):4*(i_o+offset+1)]
                        tubelets_boxes = tubelets[:, 4*i_o:4*i_o+4]
   
                        # add their IoU
                        ious += iou2d(tubelets_boxes, tube_box)

                     ious /= nov
                  else: # there is no overlap
                     # compute IoU between last box of the tube and tubelet's first boxes
                     tube_last_box = last_tubelet[4*K-4:4*K]
                     first_boxes = tubelets[:, :4]
                     ious = iou2d(first_boxes, tube_last_box)

                  valid = np.where(ious >= self.merge_iou)[0]

                  if valid.size > 0:
                     # take tubelet with maximum score
                     _maxsc = np.argmax(tubelets[valid, -1])
                     idx = valid[_maxsc] 
                     cur_tubes[i_t].append( (frame, tubelets[idx, :]) )
                     tubelets = np.delete(tubelets, idx, axis=0)
                  else:
                     if offset >= self.offset_end:
                        finished.append(i_t) 

               if self.hasShots and last_of_shot:
                  # finish all tubes
                  finished_tubes += cur_tubes
                  cur_tubes = []
               else:
                  # finish tubes
                  for i_t in finished[::-1]:
                     finished_tubes.append(cur_tubes[i_t][:])
                     del cur_tubes[i_t]

                  # start new tubes from remaing tubelets
                  for i in xrange(tubelets.shape[0]):
                     cur_tubes.append( [(frame, tubelets[i, :])] )

            # add last current tubes to finished ones
            finished_tubes += cur_tubes

            # build final tubes
            output = []
            for tube in finished_tubes:
               score = tubescore(tube)

               if score < self.min_tube_score:
                  continue

               st_frame = tube[0][0]
               en_frame = tube[-1][0] + K - 1
               tlen = en_frame - st_frame + 1
               
               # delete short tubes
               if tlen < self.min_tube_length:
                  continue

               out = np.zeros((tlen, 6), dtype=np.float32) # frame / box / score
               out[:, 0] = np.arange(st_frame, en_frame + 1)
               n_per_frame = np.zeros((tlen), dtype=np.int32)

               # average tubelets per frame
               for i in xrange( len(tube) ): # for each tube's tubelet
                  frame, box = tube[i] # get tubelet
                  i_f = frame - st_frame # frame offset in the final tube
                  for k in xrange(K):
                     out[i_f + k, 1:5] += box[4*k:4*k+4]
                     out[i_f + k, -1] += box[-1]
                     n_per_frame[i_f + k] += 1

               miss_idx = n_per_frame == 0
               valid_idx = np.logical_not(miss_idx)
               out[valid_idx, 1:] /= n_per_frame[valid_idx, None]

               if miss_idx.any():
                  # fill eventual missing detections
                  self.fillmissing(out, miss_idx)

               output.append((out, score))

            res[c] = output

         with open(outfile, 'wb') as f:
            pickle.dump(res, f)

if __name__ == '__main__':
   K = 5
   dset = 'rgb_trainall' # 'rgb_valall' 'rgb_trainall'
   dataname = 'UCF101' # 'DALY' 'UCF101'

   if dataname == 'DALY':
      proot = '/sequoia/data2/gcheron/DALY'
      pref = 'daly'
      nclasses = 10
      hasShots = True # there are several shots in a video
   elif dataname == 'UCF101':
      proot = '/sequoia/data2/gcheron/UCF101/detection'
      pref = 'ucf101'
      nclasses = 24
      hasShots = False

   dset = '%s_%s' % (pref, dset)
   if K > 1:
      dset += 'K%d' % K
   tb = tube_builder(
                     '../../output/res101/%s/faster_rcnn_10/' % dset,
                     '%s/mytracksK%d_FasterOut/' % (proot, K),
                     nclasses, K,
                     True, hasShots)
   tb.build_tubes()
