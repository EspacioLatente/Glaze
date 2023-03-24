import os
import sys

import math
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
import glaze
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
import multiprocessing
import psutil
import time
import requests
import numpy as np
import textwrap
os.environ['QT_MAC_WANTS_LAYER'] = '1'
VERSION = '1'

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def check_memory():
    total_memory = psutil.virtual_memory().total / 1073741824
    available = psutil.virtual_memory().available / 1073741824
    if total_memory < 9:
        e = 'Your computer may not have enough memory to run Glaze. \nGlaze needs around 5Gb of RAM itself. You can \nstill glaze your artwork but it may take a much longer time. \nYou can close other applications to free up memory for Glaze. '
        return e
    if available < 5.3:
        e = 'Your computer currently do not have enough memory to \nrun Glaze. Please close other apps and try again. '
        return e


def check_glaze_update():
    print('Check update')
    rep = requests.get('http://glaze.cs.uchicago.edu/software/glaze_version.json')
    up_to_date_version = int(rep.text.strip('\n').split('_')[-1])
    if int(VERSION) < up_to_date_version:
        return 1
    return None


class Worker(QThread):
    csignal = pyqtSignal('PyQt_PyObject')
    
    def __init__(self):
        QThread.__init__(self)
        self.msg = None

    
    def run(self):
        my_glaze = glaze.Glaze(0, '1', None, mode=None, opt_mode=None, output_dir=None)
        my_glaze.print(self.msg)
        intensity = self.msg['intensity']
        rq = self.msg['rq']
        output_dir = self.msg['output_dir']
        
        my_glaze.update_params(output_dir, intensity, rq, self.csignal)
        image_paths = self.msg['img_paths']
        
        try:
            res_path = my_glaze.run_protection_prod(image_paths)
            self.csignal.emit('done:{}={}'.format(rq, res_path[0]))
        except Exception as e:
            try:
                self.csignal.emit('error={}'.format(e))
            finally:
                e = None
                del e
        finally:
            del my_glaze

        return None



class Downloader(QThread):
    signal = pyqtSignal('PyQt_PyObject')
    
    def __init__(self):
        QThread.__init__(self)

        
    def run(self):
        try:
            glaze.download_all_resources(self.signal)
            self.signal.emit('done')
        except Exception as e:
            try:
                self.signal.emit('error')
                print(e)
            finally:
                e = None
                del e
        finally:
            return None



class GlazeAPP(object):
    
    def __init__(self, Form):
        uic.loadUi(resource_path('glaze.ui'), Form)
        Form.setObjectName('Form')
        Form.resize(860, 648)
        Form.setStyleSheet('background-color: #2f3136;')
        self.Form = Form
        self.running = False
        self.have_warned = False
        self.reminder = None
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(20, 40, 80, 30))
        self.pushButton.setStyleSheet('border-width: 1px;\n        padding: 1px;\n        border-style: solid;\n        border: 0px;\n        border-radius: 10px;\n        background: #629CF8;\n        outline-offset: 4px;\n        color:white ')
        self.clearButton = QtWidgets.QPushButton(Form)
        self.clearButton.setGeometry(QtCore.QRect(20, 80, 80, 30))
        self.clearButton.setStyleSheet('border-width: 1px;\n        padding: 1px;\n        border-style: solid;\n        border-radius: 10px;\n                border: 0px;\n\n        background: #4D4D55;\n        outline-offset: 4px;\n        color:white ')
        self.outputButton = QtWidgets.QPushButton(Form)
        self.outputButton.setGeometry(QtCore.QRect(20, 525, 80, 30))
        self.outputButton.setStyleSheet('border-width: 1px;\n        padding: 1px;\n        border-style: solid;\n        border-radius: 10px;\n                border: 0px;\n        background: #4D4D55;\n        outline-offset: 4px;\n        color:white ')
        self.cloakButton = QtWidgets.QPushButton(Form)
        self.cloakButton.setGeometry(QtCore.QRect(190, 585, 110, 34))
        self.cloakButton.setStyleSheet('background-color: #9C65F7;border-radius: 15px;color:white ')
        self.previewButton = QtWidgets.QPushButton(Form)
        self.previewButton.setGeometry(QtCore.QRect(60, 585, 110, 34))
        self.previewButton.setStyleSheet('background-color: #6B56F6;border-radius: 15px;color:white ')
        self.img_paths = []
        self.output_dir = None
        self.labelA = QtWidgets.QLabel(Form)
        self.labelA.setText('Waiting to load resources... ')
        self.labelA.setStyleSheet('color:white ')
        self.labelA.move(130, 60)
        self.labelMsg = QtWidgets.QLabel(Form)
        self.labelMsg.resize(400, 300)
        self.labelMsg.setAlignment(QtCore.Qt.AlignCenter)
        self.labelMsg.setStyleSheet('background-color: #202020;border-radius: 15px; color: white;font-size: 15px')
        self.labelMsg.setOpenExternalLinks(True)
        self.labelMsg.setText('Welcome to Glaze!\n\n To Glaze your work, follow\n the three step process on the left panel. ')
        self.labelMsg.move(360, 60)
        self.labelDownload = QtWidgets.QLabel(Form)
        self.labelDownload.resize(400, 100)
        self.labelDownload.setAlignment(QtCore.Qt.AlignCenter)
        self.labelDownload.setStyleSheet('color: white;font-size: 15px')
        self.labelDownload.setOpenExternalLinks(True)
        self.labelDownload.setText('Checking resources... ')
        self.labelDownload.move(360, 450)
        self.labelOutput = ScrollLabel(Form)
        self.labelOutput.resize(250, 25)
        self.labelOutput.setFont(QFont('', 12))
        self.labelOutput.setText('not select')
        self.labelOutput.move(10, 555)
        self.labelOutput.setStyleSheet('border: 0px;color:white;font-size: 13px ')
        self.pushButton.setObjectName('pushButton')
        self.outputButton.setObjectName('outButton')
        self.sl_intensity = QSlider(Qt.Horizontal, Form)
        self.sl_intensity.resize(280, 50)
        self.sl_intensity.setMinimum(0)
        self.sl_intensity.setMaximum(100)
        self.sl_intensity.setValue(25)
        self.sl_intensity.setTickPosition(QSlider.TicksBelow)
        self.sl_intensity.setTickInterval(25)
        self.sl_intensity.move(13, 240)
        self.label1 = QtWidgets.QLabel(Form)
        self.label1.setText('Very\nLow')
        self.label1.setStyleSheet('font-size: 13px;color: white')
        self.label1.move(6, 270)
        self.label2 = QtWidgets.QLabel(Form)
        self.label2.setText('Low')
        self.label2.setStyleSheet('font-size: 13px;color: white')
        self.label2.move(75, 270)
        self.label3 = QtWidgets.QLabel(Form)
        self.label3.setText('Mid')
        self.label3.setStyleSheet('font-size: 13px;color: white')
        self.label3.move(145, 270)
        self.label4 = QtWidgets.QLabel(Form)
        self.label4.setText('High')
        self.label4.setStyleSheet('font-size: 13px;color: white')
        self.label4.move(210, 270)
        self.label5 = QtWidgets.QLabel(Form)
        self.label5.setText('Very\nHigh')
        self.label5.setStyleSheet('font-size: 13px;color: white')
        self.label5.move(275, 270)
        self.intensity_text = ScrollLabel(Form)
        self.intensity_text.resize(300, 70)
        self.intensity_text.setFont(QFont('', 11))
        self.intensity_text.move(10, 175)
        self.intensity_text.setStyleSheet('border: 0px;color:#c3cba4;font-size: 14px; font: Arial')
        self.intensity_text.setFont(QFont('Arial', 15))
        self.intensity_text.setText('Magnitude of changes that will add to your art. Higher values can lead to more visible changes but stronger protection against AI. ')
        self.rq_intensity = QSlider(Qt.Horizontal, Form)
        self.rq_intensity.resize(280, 50)
        self.rq_intensity.setMinimum(0)
        self.rq_intensity.setMaximum(2)
        self.rq_intensity.setValue(1)
        self.rq_intensity.setTickPosition(QSlider.TicksBelow)
        self.rq_intensity.setTickInterval(1)
        self.rq_intensity.move(13, 415)
        self.label7 = QtWidgets.QLabel(Form)
        self.label7.setText('Faster\n(~20 mins)')
        self.label7.setStyleSheet('font-size: 13px;color: white')
        self.label7.move(6, 450)
        self.label8 = QtWidgets.QLabel(Form)
        self.label8.setText('Medium\n(~40 mins)')
        self.label8.setStyleSheet('font-size: 13px;color: white')
        self.label8.move(125, 450)
        self.label9 = QtWidgets.QLabel(Form)
        self.label9.setText('Slower\n(~60 mins)')
        self.label9.setStyleSheet('font-size: 13px;color: white')
        self.label9.move(260, 450)
        self.rd_text = ScrollLabel(Form)
        self.rd_text.resize(300, 70)
        self.rd_text.setFont(QFont('', 20))
        self.rd_text.setText('Duration spent glazing the art. Higher can leads to better protection but longer rendering time. ')
        self.rd_text.move(10, 350)
        self.rd_text.setStyleSheet('border: 0px;color:#c3cba4;font-size: 14px; font: Arial')
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.thread = Worker()
        self.thread.csignal.connect(self.finished)
        self.downloader_thread = Downloader()
        self.downloader_thread.signal.connect(self.download_handler)
        self.download_resource()
        self.check_update()

    
    def download_resource(self):
        self.cloakButton.setEnabled(False)
        self.previewButton.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.downloader_thread.start()

    
    def download_handler(self, result):
        print(result)
        if result.startswith('download'):
            msg = result.split('=')[-1]
            self.labelDownload.setText(msg)
            self.labelDownload.repaint()
        if result.startswith('done'):
            self.labelDownload.setText('Resource loaded successfully. ')
            self.labelA.setText('Select image(s) to Glaze. ')
            self.labelDownload.repaint()
            self.labelA.repaint()
            self.cloakButton.setEnabled(True)
            self.previewButton.setEnabled(True)
            self.pushButton.setEnabled(True)
        if result.startswith('error'):
            self.labelDownload.setText('Error loading resources, please \nrestart the app to try again. ')
            self.labelDownload.repaint()
            return None

    
    def check_update(self):
        is_update = check_glaze_update()
        if is_update:
            self.labelMsg.setText("A new version of Glaze is available. (<a href='http://glaze.cs.uchicago.edu/'>Download Link</a>)")
            return None

    
    def retranslateUi(self, Form):
        self.tr = QtCore.QCoreApplication.translate
        Form.setWindowTitle(self.tr('Form', 'Glaze - Protecting artists from invasive AI'))
        self.pushButton.setText(self.tr('Form', 'Select...'))
        self.clearButton.setText(self.tr('Form', 'Clear All'))
        self.outputButton.setText(self.tr('Form', 'Save As...'))
        self.cloakButton.setText(self.tr('Form', 'Run Glaze'))
        self.previewButton.setText(self.tr('Form', 'Preview'))
        self.pushButton.clicked.connect(self.pushButton_handler)
        self.clearButton.clicked.connect(self.clearButton_handler)
        self.outputButton.clicked.connect(self.outputButton_handler)
        None((lambda : self.protect_images()))
        None((lambda : self.protect_images(True, **('preview',))))

    
    def pushButton_handler(self):
        print('Button pressed')
        self.open_dialog_box()

    
    def clearButton_handler(self):
        self.img_paths = []
        self.labelA.setText('Selected {} images'.format(len(self.img_paths)))

    
    def outputButton_handler(self):
        print('Button pressed')
        qfd = QFileDialog()
        output_dir = QFileDialog.getExistingDirectory(qfd, 'Select Output Folder')
        self.output_dir = output_dir
        print('Selected output dir', self.output_dir)
        self.labelOutput.setText('{}'.format(self.output_dir))

    
    def open_dialog_box(self):
        qfd = QFileDialog()
        path = '.'
        filter = 'Images (*.png *.xpm *.jpg *jpeg *.gif)'
        filename = QFileDialog.getOpenFileNames(qfd, 'Select Image(s)', path, filter)
        self.img_paths += filename[0]
        print('Selected paths', self.img_paths)
        self.labelA.setText('Selected {} images'.format(len(self.img_paths)))

    
    def finished(self, result):
        print(result)
        if result.startswith('glazetp'):
            cur_percentage = result.split('=')[-1]
            res = self.reminder.get_reminder(float(cur_percentage))
        if res is not None:
            res = int(res)
            (m, s) = divmod(res, 60)
            self.labelMsg.setText('Glazing ~{} mins {} secs left'.format(m, s))
            self.labelMsg.repaint()
        if result.startswith('display'):
            msg = result.split('=')[-1]
            print('update', msg)
            msg = textwrap.fill(msg, 40)
            self.labelMsg.setText(msg)
            self.labelMsg.repaint()
        if result.startswith('error'):
            msg = result.split('=')[-1]
            print('update', msg)
            msg = textwrap.fill(msg, 40)
            self.labelMsg.setText(msg)
            self.labelMsg.repaint()
            self.cloakButton.setEnabled(True)
            self.previewButton.setEnabled(True)
        if result.startswith('done'):
            res_path = result.split('=')[-1]
            is_preview = result.split('=')[0].split(':')[-1] == '-1'
            if is_preview:
                self.labelMsg.linkActivated.connect(self.link)
                self.labelMsg.setText('<a href="file:///{}">Click to Preview/</a>'.format(res_path))
            else:
                self.labelMsg.setText('Glaze succeed, glazed images saved at your output folder')
                self.img_paths = []
                self.labelA.setText('Select image(s) to Glaze')
            self.labelMsg.repaint()
            self.cloakButton.setEnabled(True)
            self.previewButton.setEnabled(True)
            self.pushButton.setEnabled(True)
            self.reminder = None
            return None

    
    def link(self, linkStr):
        QDesktopServices.openUrl(QUrl(linkStr))

    
    def protect_images(self, preview = (False,)):
        if not self.have_warned:
            mem_res = check_memory()
            if mem_res != 0:
                self.labelMsg.setText(mem_res)
                self.labelMsg.repaint()
                self.have_warned = True
                return None
            if None(self.img_paths) == 0:
                self.labelMsg.setText('Please select images first.')
                return None
            if None.output_dir is None:
                self.labelMsg.setText('Please select a output folder. ')
                return None
            self.reminder = None()
            cur_intensity = self.sl_intensity.value()
            cur_rq = self.rq_intensity.value()
            if preview:
                cur_rq = -1
                cur_output_dir = self.output_dir
        self.img_paths = list(set(self.img_paths))
        msg = {
            'img_paths': self.img_paths,
            'intensity': cur_intensity,
            'rq': cur_rq,
            'output_dir': cur_output_dir }
        self.thread.msg = msg
        self.cloakButton.setEnabled(False)
        self.previewButton.setEnabled(False)
        self.thread.start()



class ScrollLabel(QtWidgets.QScrollArea):
    '''ScrollLabel'''
    
    def __init__(self, *args, **kwargs):
        super(ScrollLabel, self).__init__(*args, **kwargs)
        self.setWidgetResizable(True)
        
        content = QtWidgets.QWidget(self)
        self.setWidget(content)
        
        lay = QtWidgets.QVBoxLayout(content)
        
        self.label = QtWidgets.QLabel(content)
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        
        lay.addWidget(self.label)

    
    def setText(self, text):
        self.label.setText(text)



class ReminderClock(object):
    
    def __init__(self):
        self.time_queue = []
        self.last_time = None
        self.prev = 0

    
    def get_reminder(self, cur_p):
        cur_time = time.time()
        if self.last_time is None:
            self.last_time = cur_time
            self.prev = cur_p
            return None
        cur_time_gap = None - self.last_time
        self.time_queue.append(cur_time_gap)
        median_time = np.median(self.time_queue)
        incr = cur_p - self.prev
        reminder_time = (median_time / incr) * (100 - cur_p)
        self.prev = cur_p
        self.last_time = cur_time
        if len(self.time_queue) > 2:
            return reminder_time



# def protect_pytransform():
#
#     def assert_builtin(func):
#         type = ''.__class__.__class__
#         builtin_function = type(''.join)
#         if type(func) is not builtin_function:
#             raise RuntimeError('%s() is not a builtin' % func.__name__)
#
#
#     def check_obfuscated_script():
#         _getframe = _getframe
#         import sys
#         CO_SIZES = (30, 39)
#         CO_NAMES = set([
#             'pytransform',
#             'pyarmor',
#             '__name__',
#             '__file__'])
#         co = _getframe(3).f_code
#         if not set(co.co_names) <= CO_NAMES or len(co.co_code) in CO_SIZES:
#             raise RuntimeError('unexpected obfuscated script')

    
    # def check_lib_pytransform():
    #     platform = platform
    #     import sys
    #     if platform == 'darwin':
    #         return None
    #     import pytransform
    #     filename = pytransform.__file__
    #     with open(filename, 'rb') as f:
    #         buf = bytearray(f.read())
    #         None(None, None, None)


    # assert_builtin(sum)
    # assert_builtin(open)
    # assert_builtin(len)
    # check_obfuscated_script()
    # check_lib_pytransform()

# protect_pytransform()
if __name__ == '__main__':
    import sys
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = GlazeAPP(Form)
    Form.show()
    sys.exit(app.exec_())

