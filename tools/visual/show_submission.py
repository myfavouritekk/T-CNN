import numpy as np
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from functools import partial
from collections import defaultdict
from easydict import EasyDict
from PyQt4 import QtGui, QtCore


DATA_DIR = 'External/exp/dataset/ILSVRC2015/Data/VID'
IMAGESETS_DIR = 'External/exp/dataset/ILSVRC2015/ImageSets/VID'
SUBMISSION_DIR = 'External/exp/result_vid'
SCREENSHOT_DIR = 'External/exp/screenshot'
VIDEOSHOT_DIR = 'External/exp/videoshot'


CLASS_NAMES = ['background', 'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']
PRESET_COLORS = [
  QtGui.QColor(255, 65, 54),
  QtGui.QColor(61, 153, 112),
  QtGui.QColor(0, 116, 217),
  QtGui.QColor(133, 20, 75),
  QtGui.QColor(0, 31, 63),
  QtGui.QColor(240, 18, 190),
  QtGui.QColor(1, 255, 112),
  QtGui.QColor(127, 219, 255),
  QtGui.QColor(255, 133, 27),
  QtGui.QColor(176, 176, 176)
]


def read_submission(file_path, subset):
    videos = defaultdict(list)
    fid_to_path = {}
    with open(osp.join(IMAGESETS_DIR, subset + '.txt')) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            fid_to_path[int(line[1])] = osp.join(
                DATA_DIR, subset, line[0] + '.JPEG')
            videos[osp.dirname(line[0])].append(int(line[1]))
    for k in videos:
        videos[k].sort()

    ret = defaultdict(list)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            item = {
                'fid': int(line[0]),
                'class_index': int(line[1]),
                'score': float(line[2]),
                'bbox': map(float, line[3:])
            }
            item = EasyDict(item)
            ret[item.fid].append(item)
    return videos, fid_to_path, ret


def draw_predictions(file_path, predictions, class_index,
                     score_low, score_high):
    img = QtGui.QImage(file_path)
    painter = QtGui.QPainter(img)
    for i, pred in enumerate(predictions):
        if class_index > 0 and pred.class_index != class_index: continue
        if pred.score < score_low or pred.score > score_high: continue
        class_name = CLASS_NAMES[pred.class_index]
        x1, y1, x2, y2 = map(int, pred.bbox)
        # bbox
        painter.setPen(QtGui.QPen(PRESET_COLORS[i % len(PRESET_COLORS)], 10.0))
        painter.setBrush(QtGui.QBrush())
        painter.drawRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)
        # label background rect
        painter.setPen(QtGui.QPen(PRESET_COLORS[i % len(PRESET_COLORS)], 2.0))
        painter.setBrush(QtGui.QBrush(PRESET_COLORS[i % len(PRESET_COLORS)]))
        if class_index > 0:
            painter.drawRect(x1, y1, min(x2 - x1 + 1, 100), 30)
        else:
            painter.drawRect(x1, y1, min(x2 - x1 + 1, 200), 30)
        # label text
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.setBrush(QtGui.QBrush())
        painter.setFont(QtGui.QFont('Arial', 20, QtGui.QFont.Bold))
        if class_index > 0:
            painter.drawText(x1 + 4, y1 + 24, '{:.2f}'.format(pred.score))
        else:
            painter.drawText(x1 + 4, y1 + 24,
                             '{} {:.2f}'.format(class_name, pred.score))
    return img


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.build_ui()

    def build_ui(self):
        # menu
        file_menu = self.menuBar().addMenu('&File')
        file_menu.addAction('Open val', partial(self.open, subset='val'))
        file_menu.addAction('Open test', partial(self.open, subset='test'))
        # toolbar
        next_act = QtGui.QAction('Next', self)
        next_act.setShortcut('d')
        next_act.triggered.connect(self.show_next)
        prev_act = QtGui.QAction('Prev', self)
        prev_act.setShortcut('a')
        prev_act.triggered.connect(self.show_prev)
        self.status = QtGui.QLabel()
        self.jumpto = QtGui.QLineEdit()
        self.jumpto.setFixedWidth(50)
        self.jumpto.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.jumpto.returnPressed.connect(self.jump)
        self.total = QtGui.QLabel()
        self.combobox = QtGui.QComboBox()
        self.combobox.addItems(['all'] + CLASS_NAMES[1:])
        self.combobox.currentIndexChanged[int].connect(self.show_class)
        self.cur_class_index = 0
        self.score_low_edit = QtGui.QLineEdit('-100')
        self.score_low_edit.setFixedWidth(60)
        self.score_high_edit = QtGui.QLineEdit('100')
        self.score_high_edit.setFixedWidth(60)
        self.score_low = -100
        self.score_high = 100
        score_button = QtGui.QPushButton("Set")
        score_button.setFixedWidth(50)
        score_button.clicked.connect(self.set_score_range)
        screenshot_act = QtGui.QAction('Screenshot', self)
        screenshot_act.triggered.connect(self.screenshot)
        videoshot_act = QtGui.QAction('Videoshot', self)
        videoshot_act.triggered.connect(self.videoshot)
        toolbar = self.addToolBar("Tool bar")
        toolbar.addWidget(self.status)
        toolbar.addAction(prev_act)
        toolbar.addAction(next_act)
        toolbar.addWidget(QtGui.QLabel("  Jump to: "))
        toolbar.addWidget(self.jumpto)
        toolbar.addWidget(self.total)
        toolbar.addWidget(QtGui.QLabel("  Category: "))
        toolbar.addWidget(self.combobox)
        toolbar.addWidget(QtGui.QLabel("     Score range: "))
        toolbar.addWidget(self.score_low_edit)
        toolbar.addWidget(QtGui.QLabel(" - "))
        toolbar.addWidget(self.score_high_edit)
        toolbar.addWidget(score_button)
        toolbar.addAction(screenshot_act)
        toolbar.addAction(videoshot_act)
        # video list
        self.video_list = QtGui.QListWidget()
        self.video_list.setSizePolicy(QtGui.QSizePolicy.Maximum,
                QtGui.QSizePolicy.Ignored)
        self.video_list.itemDoubleClicked.connect(self.show_video)
        # image panel
        self.image_panel = QtGui.QLabel()
        self.image_panel.setSizePolicy(QtGui.QSizePolicy.Ignored,
                QtGui.QSizePolicy.Ignored)
        self.image_panel.setScaledContents(True)
        self.image_panel.setAlignment(QtCore.Qt.AlignCenter)
        # layout
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.video_list)
        layout.addWidget(self.image_panel)
        # main frame
        frame = QtGui.QWidget()
        frame.setLayout(layout)
        self.setCentralWidget(frame)
        self.setWindowTitle('VID submission viewer')
        self.showMaximized()

    def refresh_ui(self):
        self.video_list.clear()
        self.video_list.addItems(sorted(self.videos.keys()))
        self.image_panel.clear()
        self.status.clear()
        self.jumpto.clear()
        self.total.clear()

    def open(self, subset='val'):
        file_path = QtGui.QFileDialog.getOpenFileName(self,
                'Open {} submission'.format(subset), SUBMISSION_DIR)
        if len(str(file_path)) == 0: return
        # Read submission text
        self.videos, self.fid_to_path, self.ret = read_submission(
                file_path, subset)
        self.refresh_ui()
        self.setWindowTitle(osp.splitext(osp.basename(str(file_path)))[0])

    def show_video(self, item):
        vid_name = str(item.text())
        self.vid_name = vid_name
        self.fids = self.videos[vid_name]
        self.frames = [self.fid_to_path[fid] for fid in self.fids]
        self.show_frame(0)
        status_text = vid_name
        self.status.setText(status_text)
        self.total.setText(' / {}  '.format(len(self.frames)))

    def show_frame(self, frame_index):
        self.cur_frame_index = frame_index
        file_path = self.frames[frame_index]
        # image with predictions
        preds = self.ret[self.fids[frame_index]]
        qimg = draw_predictions(file_path, preds,
                                self.cur_class_index,
                                self.score_low, self.score_high)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.image_panel.setPixmap(pixmap)
        # status
        self.jumpto.setText(str(frame_index + 1))
        self.jumpto.clearFocus()

    def show_prev(self):
        if self.cur_frame_index > 0:
            self.show_frame(self.cur_frame_index - 1)

    def show_next(self):
        if self.cur_frame_index + 1 < len(self.frames):
            self.show_frame(self.cur_frame_index + 1)

    def jump(self):
        index = int(self.jumpto.text()) - 1
        if 0 <= index and index < len(self.frames):
            self.show_frame(index)

    def show_class(self, index):
        self.cur_class_index = index
        self.show_frame(self.cur_frame_index)

    def set_score_range(self):
        try:
            low = float(self.score_low_edit.text())
            high = float(self.score_high_edit.text())
            if low > high: raise ValueError()
        except:
            QtGui.QMessageBox.warning(self, 'Invalid score range',
                    'Score should be real number, and low <= high',
                    QtGui.QMessageBox.Ok)
            return
        self.score_low = low
        self.score_high = high
        self.show_frame(self.cur_frame_index)

    def screenshot(self):
        file_name = osp.splitext(osp.basename(self.frames[self.cur_frame_index]))[0]
        file_name = self.vid_name + '_' + file_name + '.png'
        pixmap = self.image_panel.pixmap()
        pixmap.save(osp.join(SCREENSHOT_DIR, file_name))

    def videoshot(self):
        folder = osp.join(VIDEOSHOT_DIR, self.vid_name)
        if not osp.isdir(folder):
            os.makedirs(folder)
        for frame_index, frame_path in enumerate(self.frames):
            preds = self.ret[self.fids[frame_index]]
            qimg = draw_predictions(frame_path, preds, self.cur_class_index,
                                    self.score_low, self.score_high)
            file_name = osp.splitext(osp.basename(frame_path))[0]
            file_name = self.vid_name + '_' + file_name + '.png'
            qimg.save(osp.join(folder, file_name))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())