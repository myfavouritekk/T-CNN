import caffe
import yaml
import json
import numpy as np
import os
import random
import gzip

def proto_load(file_path):
    # load .gz version if exists
    # AD_HOC implementation
    if os.path.isfile(file_path + '.gz'):
        file_path += '.gz'
    if os.path.splitext(file_path)[1] == '.gz':
        with gzip.GzipFile(file_path) as f:
            obj = json.loads(f.read())
    else:
        with open(file_path, 'r') as f:
            obj = json.load(f)
    return obj


class TrackDataLayer(caffe.Layer):
    """Track data layer for providing track score and features"""

    def _track_preprocess(self):
        tracks = []
        for score_file in self._score_files:
            score_proto = proto_load(score_file)
            for tubelet in score_proto['tubelets']:
                track = {}
                # general information
                track['length'] = len(tubelet['boxes'])
                track['gt'] = tubelet['gt']
                track['mean_iou'] = np.mean([map(lambda x:x['gt_overlap'],
                                          tubelet['boxes'])])

                # tops
                track['det_scores'] = map(lambda x:x['det_score'],
                                          tubelet['boxes'])
                track['track_scores'] = map(lambda x:x['track_score'],
                                          tubelet['boxes'])
                track['anchors'] = map(lambda x:x['anchor'] * 1. / track['length'],
                                          tubelet['boxes'])
                track['abs_anchors'] = map(abs, track['anchors'])
                track['gt_overlaps'] = map(lambda x:x['gt_overlap'],
                                          tubelet['boxes'])
                track['labels'] = [1 if iou >= 0.5 else 0 for iou in track['gt_overlaps']]
                # skip memory heavy features if possible
                if 'all_scores' in self._top_names:
                    track['all_scores'] = map(lambda x:x['all_score'],
                                          tubelet['boxes'])
                if 'feats' in self._top_names:
                    track['feats'] = map(lambda x:x['feat'],
                                          tubelet['boxes'])
                tracks.append(track)
        return tracks

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        config = yaml.load(open(layer_params['config']).read())

        assert len(top) >= 1 and len(top) <= 6
        self._tot_top_names = [
            'det_scores',
            'track_scores',
            'abs_anchors',
            'all_scores',
            'feats'
        ]

        # phase, current only support setting phase in config file, so be careful
        try:
            self._phase = config['phase']
            assert self._phase in ['test', 'train', 'valid']
        except KeyError:
            # default phase is test
            self._phase = 'test'
        self._top_names = self._tot_top_names[0:len(top)-1]
        self._name_to_top_map = dict(zip(self._top_names,xrange(len(self._top_names))))
        self._name_to_top_map['labels'] = len(top)-1
        self._length = config['length']
        self._batch_size = config['batch_size']
        self._pos_ratio = config['pos_ratio']
        self._num_pos = int(self._pos_ratio * self._batch_size)
        self._score_files = map(lambda x: os.path.join(config['root'], x.strip()),
                                open(config['source']))
        self._tracks = self._track_preprocess()
        self._mean_ious = map(lambda x:x['mean_iou'], self._tracks)
        self._pos_index = [i for i, iou in enumerate(self._mean_ious) if iou >= 0.5]
        self._neg_index = [i for i in xrange(len(self._mean_ious)) if i not in self._pos_index]
        self._track_index = range(len(self._tracks))

        # reshape top blobs
        for top_id, top_name in zip(xrange(len(top)), self._top_names + ['labels']):
            sample_feat = self._tracks[0][top_name][0]
            if type(sample_feat) is list:
                feat_length = len(sample_feat)
            else:
                feat_length = 1
            top[top_id].reshape(self._batch_size, feat_length, 1, self._length)


    def _rotate_list(self, l, n):
        return l[n:] + l[:n]


    def forward(self, bottom, top):
        # select batch for different phases
        if self._phase == 'train':
            pos_index = np.random.choice(self._pos_index, size=self._num_pos).tolist()
            neg_index = np.random.choice(self._neg_index,
                size=self._batch_size - self._num_pos).tolist()
            batch_index = pos_index+neg_index
        elif self._phase == 'valid':
            batch_index = np.random.randint(len(self._tracks), size=self._batch_size).tolist()
        else:
            # test insequence
            batch_index = self._track_index[:self._batch_size]
            self._track_index = self._rotate_list(self._track_index, self._batch_size)

        # prepare batch
        for num_id, track_id in enumerate(batch_index):
            track = self._tracks[track_id]
            # sample starting point, 0 if track length is too short
            st_index = random.randint(0, max(0, track['length'] - self._length))
            valid_length = min(track['length'], self._length)
            for blob_name in self._top_names:
                top_id = self._name_to_top_map[blob_name]
                data = np.zeros_like(top[top_id].data[num_id])
                feat_length = data.shape[0]
                orig_data = np.asarray(track[blob_name]).reshape((feat_length, 1, -1))
                # data[:, 0, 0:valid_length] = \
                #     track[blob_name][st_index:st_index+valid_length]
                data[:, :, 0:valid_length] = orig_data[:, :, st_index:st_index+valid_length]
                top[top_id].data[num_id] = data.astype(np.float32, copy=True)
            # use -1 as ignored label
            top_label_id = self._name_to_top_map['labels']
            labels = -np.ones_like(top[top_label_id].data[num_id])
            valid_labels = np.asarray(track['labels'][st_index:st_index+valid_length])
            labels[0, 0, 0:valid_length] = valid_labels
            top[top_label_id].data[num_id] = labels.astype(np.float32, copy=True)

    def reshape(self, bottom, top):
        pass
