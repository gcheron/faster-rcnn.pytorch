import numpy as np
import pickle
import ipdb
import re
from tube_utils import iou2d
import os

class tube_builder():
   def __init__(self, detfile, resdir, nclasses, K):
      self.detfile = detfile
      self.resdir = resdir
      self.nclasses = nclasses
      self.K = K
      self.nms = 0.3 # nms on tubelets
      self.top_k = 10 # max nb of tubelets to keep at each it
      self.merge_iou = 0.2 # min iou to merge a tubelet with tube
      self.offset_end = 5 # frame number without merging tubelet after which a tube ends
      self.min_tube_score = 0.01 # min score to keep the final tube
      self.min_tube_length = 15 # min length to keep the final tube

      with open(self.detfile) as f:
         self.detections = pickle.load(f)
      assert len(self.detections) == self.nclasses + 2 # bkg + detpath

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

      # split detections per video
      prev_vid = ''
      num_dets = len(self.detections[0])
      video_dets = {}
      for i_det in xrange(num_dets):
         cur_vid = self.detections[-1][i_det]
         assert len(cur_vid) == 1
         rres = re.match('.*/([^/]*)/image-([0-9]*)\.*',cur_vid[0])
         cur_vid, cur_im = rres.group(1), int(rres.group(2))

         if cur_vid != prev_vid:
            assert not cur_vid in video_dets
            video_dets[cur_vid] = [[] for _ in xrange(self.nclasses)]
            assert cur_im == 1
         else:
            assert cur_im == prev_im + 1

         for c in xrange(self.nclasses):
            cdets = self.detections[c+1][i_det] # skip bkg
            video_dets[cur_vid][c].append(cdets)

         prev_vid = cur_vid
         prev_im = cur_im

      # link tubelets into tubes
      for i_v, vid in enumerate(video_dets):
         vdets = video_dets[vid]
         outfile = '%s/%s.pkl' % (self.resdir, vid)
         nframes = len(vdets[0]) 

         if i_v % 50 == 0:
            print '%d/%d: save %s' % (i_v + 1, len(video_dets), outfile)
         
         res = {}
         for c in xrange(self.nclasses):
            finished_tubes = []
            cur_tubes = []

            for i_f in xrange(nframes - K + 1):
               frame = i_f + 1
               # get tubelets and NMS
               tubelets = vdets[c][i_f] # get K boxes and score
               idx = self.nms_tubelets(tubelets)
               tubelets = tubelets[idx, :]

               if i_f == 0:
                  # start tubes and continue
                  for i in xrange(tubelets.shape[0]):
                     cur_tubes.append( [(1, tubelets[i, :])] )
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
   #dset = 'ucf101_rgb_valall'
   dset = 'ucf101_rgb_trainall'
   nclasses = 24
   K = 1
   tb = tube_builder('../../output/res101/%s/faster_rcnn_10/detections.pkl' % dset,
                     '../../output/res101/%s/faster_rcnn_10/tubes' % dset,
                     nclasses,
                     K)
   tb.build_tubes()
