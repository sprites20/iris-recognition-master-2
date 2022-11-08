# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'eye.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from iris_recognition import *
import os
import pickle
import time
import shutil
import glob
import pathlib

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from pathlib import Path    
import json



curr_image = ""
eyechanged = False

filepath1 = r'./aeval1.bmp'
filepath2 = r'./aeval2.bmp'

folder_pair = {}
eye_arr = []

#compare_binfiles("C:/Users/NakaMura/Downloads/iris-recognition-master (1)/iris-recognition-master/enrolledimages/1/left/1/bin.bin", "C:/Users/NakaMura/Downloads/iris-recognition-master (1)/iris-recognition-master/enrolledimages/1/left/1/bin.bin")

from numba import jit, cuda

def save_data():
    names_file = open('data/names.dat', 'wb')
    global folder_pair
    pickle.dump(folder_pair, names_file)
    names_file.close()
def load_data():
    try:
        names_df_file = open('data/names.dat', 'rb')
        global folder_pair
        folder_pair = pickle.load(names_df_file)
        names_df_file.close()
    except:
        None

currentimage = ""



def get_folders(MYDIR):
    folder_nums = []
    for entry_name in os.listdir(MYDIR):
        entry_path = os.path.join(MYDIR, entry_name)
        if os.path.isdir(entry_path):
            try:
                a = int(entry_name)
                folder_nums.append(entry_name)
            except:
                None
    return folder_nums
    #print(folder_nums)
    
folder_nums = get_folders("./enrolledimages")
load_data()
print(folder_pair)

def enrolleye(imgpath, eyename, side):
    #curreyefolder_main = get_folders("./enrolledimages")
    print(Path(imgpath).name)
    print(eyename)
    print("left" if side == 0 else "right")
    
    curreyenum = 0
    currpath = './enrolledimages'
    try:
        print(folder_pair)
        if os.path.isdir('./enrolledimages/' + str(folder_pair[eyename])):
            curreyenum = int(folder_pair[eyename])
            currpath = './enrolledimages/' + str(folder_pair[eyename])
    except:
        for i in range(1,1000000000):
            if not os.path.isdir('./enrolledimages/' + str(i)):
                curreyenum = i
                os.mkdir('./enrolledimages/' + str(i))
                currpath = './enrolledimages/' + str(i)
                folder_pair[eyename] = str(i)
                save_data()
                break
            else:
                curreyenum = i
                currpath = './enrolledimages/' + str(i)
    #Generate left/right folder
    if side == 0:
        if not os.path.isdir(currpath + '/left'):
            os.mkdir(currpath + '/left')
            currpath += '/left'
        else:
            currpath += '/left'
    elif side == 1:
        if not os.path.isdir(currpath + '/right'):
            os.mkdir(currpath + '/right')
            currpath += '/right'
        else:
            currpath += '/right'
    else:
        print("Invalid side!")
        return None
    #Generate Eye Folder
    for i in range(1,1000000000):
        if not os.path.isdir(currpath + '/' + str(i)):
            curreyenum = i
            os.mkdir(currpath + '/' + str(i))
            currpath += '/' + str(i)
            break

    if os.path.isdir(currpath):
        roi = load_rois_from_image(imgpath, currpath)
            #Cache
        enrollcache = {
            "groundtruth" : Path(imgpath).name,
            "eyename" : eyename,
            "side" : "left" if side == 0 else "right",
            "enrolldest" : currpath
        }
        y = json.dumps(enrollcache, indent=4)
        #x = json.loads(enrollcache)
        
        with open(currpath + '/' + "info.json", "w") as outfile:
            outfile.write(y)
        print(enrollcache)
        return roi
        

class ImageDropBox(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(QtCore.QRect(10, 20, 251, 261))
        self.setAcceptDrops(True)
        self.setScaledContents(True)

        #mainLayout = QVBoxLayout()

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)

            event.accept()
        else:
            event.ignore()
    def set_image(self, file_path):
        global curr_image
        curr_image = file_path
        self.setPixmap(QPixmap(file_path))
        global eyechanged
        eyechanged = True

class Ui_MainWindow(object):
    def ButtonConnections(self):
        self.pushButton.clicked.connect(self.thisenroll)
        self.pushButton_2.clicked.connect(self.masstest)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1154, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = ImageDropBox(self.centralwidget)
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 400, 121, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 400, 121, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(80, 320, 181, 31))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(10, 10, 82, 17))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(90, 10, 82, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(80, 360, 181, 31))
        self.textEdit.setObjectName("textEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 370, 61, 20))
        self.label_2.setObjectName("label_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(80, 290, 181, 31))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton_7 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_7.setGeometry(QtCore.QRect(10, 10, 82, 17))
        self.radioButton_7.setObjectName("radioButton_7")
        self.radioButton_8 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_8.setGeometry(QtCore.QRect(90, 10, 82, 17))
        self.radioButton_8.setObjectName("radioButton_8")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 330, 61, 20))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 300, 61, 20))
        self.label_5.setObjectName("label_5")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(380, 20, 61, 20))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(700, 20, 71, 20))
        self.label_15.setObjectName("label_15")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(280, 410, 261, 131))
        self.label_24.setAcceptDrops(True)
        self.label_24.setText("")
        self.label_24.setPixmap(QtGui.QPixmap("../Downloads/iris-recognition-master (1)/iris-recognition-master/left-side_matches.jpg"))
        self.label_24.setScaledContents(True)
        self.label_24.setObjectName("label_24")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(10, 450, 251, 211))
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_28 = QtWidgets.QLabel(self.centralwidget)
        self.label_28.setGeometry(QtCore.QRect(310, 380, 61, 20))
        self.label_28.setObjectName("label_28")
        self.label_30 = QtWidgets.QLabel(self.centralwidget)
        self.label_30.setGeometry(QtCore.QRect(310, 550, 71, 20))
        self.label_30.setObjectName("label_30")
        self.label_34 = QtWidgets.QLabel(self.centralwidget)
        self.label_34.setGeometry(QtCore.QRect(620, 380, 81, 20))
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.centralwidget)
        self.label_35.setGeometry(QtCore.QRect(620, 550, 81, 20))
        self.label_35.setObjectName("label_35")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(70, 670, 191, 31))
        self.textEdit_3.setObjectName("textEdit_3")
        self.label_36 = QtWidgets.QLabel(self.centralwidget)
        self.label_36.setGeometry(QtCore.QRect(10, 680, 61, 20))
        self.label_36.setObjectName("label_36")
        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(890, 50, 251, 111))
        self.textEdit_4.setObjectName("textEdit_4")
        self.label_39 = QtWidgets.QLabel(self.centralwidget)
        self.label_39.setGeometry(QtCore.QRect(280, 50, 261, 291))
        self.label_39.setAcceptDrops(True)
        self.label_39.setText("")
        self.label_39.setPixmap(QtGui.QPixmap("../Downloads/iris-recognition-master (1)/iris-recognition-master/aeval1.bmp.jpg"))
        self.label_39.setScaledContents(True)
        self.label_39.setObjectName("label_39")
        self.label_40 = QtWidgets.QLabel(self.centralwidget)
        self.label_40.setGeometry(QtCore.QRect(600, 50, 261, 291))
        self.label_40.setAcceptDrops(True)
        self.label_40.setText("")
        self.label_40.setPixmap(QtGui.QPixmap("../Downloads/iris-recognition-master (1)/iris-recognition-master/aeval2.bmp.jpg"))
        self.label_40.setScaledContents(True)
        self.label_40.setObjectName("label_40")
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(280, 570, 261, 131))
        self.label_26.setAcceptDrops(True)
        self.label_26.setText("")
        self.label_26.setPixmap(QtGui.QPixmap("../Downloads/iris-recognition-master (1)/iris-recognition-master/bottom_matches.jpg"))
        self.label_26.setScaledContents(True)
        self.label_26.setObjectName("label_26")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(600, 410, 261, 131))
        self.label_25.setAcceptDrops(True)
        self.label_25.setText("")
        self.label_25.setPixmap(QtGui.QPixmap("../Downloads/iris-recognition-master (1)/iris-recognition-master/right-side_matches.jpg"))
        self.label_25.setScaledContents(True)
        self.label_25.setObjectName("label_25")
        self.label_27 = QtWidgets.QLabel(self.centralwidget)
        self.label_27.setGeometry(QtCore.QRect(600, 570, 261, 131))
        self.label_27.setAcceptDrops(True)
        self.label_27.setText("")
        self.label_27.setPixmap(QtGui.QPixmap("../Downloads/iris-recognition-master (1)/iris-recognition-master/complete_matches.jpg"))
        self.label_27.setScaledContents(True)
        self.label_27.setObjectName("label_27")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(980, 20, 71, 20))
        self.label_16.setObjectName("label_16")
        self.textEdit_5 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_5.setGeometry(QtCore.QRect(890, 210, 251, 111))
        self.textEdit_5.setObjectName("textEdit_5")
        self.textEdit_4.setReadOnly(True)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(980, 180, 71, 20))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(980, 340, 71, 20))
        self.label_18.setObjectName("label_18")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.ButtonConnections()
    def load_matches(self):
        #self.label_39.setPixmap(QtGui.QPixmap("./tempeye/complete.jpg"))
        #self.label_40.setPixmap(QtGui.QPixmap("C:/Users/NakaMura/Downloads/iris-recognition-master (1)/iris-recognition-master/enrolledimages/1/left/1/complete.jpg"))
        self.label_24.setPixmap(QtGui.QPixmap('./left-side_matches.jpg'))
        self.label_25.setPixmap(QtGui.QPixmap('./right-side_matches.jpg'))
        self.label_26.setPixmap(QtGui.QPixmap('./bottom_matches.jpg'))
        self.label_27.setPixmap(QtGui.QPixmap('./complete_matches.jpg'))
        #time.sleep(2)
    def get_folders(self, MYDIR):
        folder_nums = []
        for entry_name in os.listdir(MYDIR):
            entry_path = os.path.join(MYDIR, entry_name)
            if os.path.isdir(entry_path):
                try:
                    a = int(entry_name)
                    folder_nums.append(entry_name)
                except:
                    None
        return folder_nums
        #print(folder_nums)
    #@jit(target_backend='cuda')
    def recognize(self, image_curr, side):
        print(image_curr)
        #Get list of eyes in same side
        folder_nums = self.get_folders("./enrolledimages")
        folder_num2 = {}
        #print(folder_nums)
        sidetext = "left" if side==0 else "right" if side==1 else "left"
        for i in folder_nums:
            folder_num2[i] = self.get_folders("./enrolledimages/" + i + '/' + sidetext)
            #print(folder_num2[i])
        try:
            shutil.rmtree('./tempeye/')
            os.mkdir('./tempeye')
        except:
            try:
                os.mkdir('./tempeye')
            except:
                None
        rois, main_keypoints = load_rois_from_image(image_curr, './tempeye')
        #rois = load_rois_from_image(curr_image, './tempeye')
        self.label_39.setPixmap(QtGui.QPixmap("./tempeye/equalized histogram iris region.jpg"))
        
        best_match_array = {}
        best_match_value = 0
        best_match_path = ""
        best_match_string = ""
        first_match = True
        rois1cache = None
        keycache1cache = None
        key, matches, rois1, keycache1 = None, None, None, None
        for i in folder_nums:
            for j in folder_num2[i]:
                curr_match_value = 0
                data = json.load(open("./enrolledimages/" + i + '/' + sidetext + '/' + j + "/info.json"))
                ground = data["groundtruth"].split('_')
                currmg = image_curr.split('/')[-1].split('_')
                #print(currmg)
                #print(ground)
                if ground[-2] == currmg[-2] and ground[-3] == currmg[-3]:
                    print(data["groundtruth"])
                    if first_match:
                        key, matches, rois1, keycache1 = compare_binfiles('./tempeye/bin.bin', "./enrolledimages/" + i + '/' + sidetext + '/' + j + '/bin.bin', None, None)
                        rois1cache = rois1
                        keycache1cache = keycache1
                        first_match = False
                    else:
                        key, matches = compare_binfiles('./tempeye/bin.bin', "./enrolledimages/" + i + '/' + sidetext + '/' + j + '/bin.bin', rois1cache, keycache1cache)
                    #print(key)
                    #print(matches)
                    for pos in ['right-side','left-side','bottom','complete']:
                        curr_match_value += matches[pos]/main_keypoints[pos]/4
                    if curr_match_value >= best_match_value:
                        best_match_value = curr_match_value
                        best_match_path = "./enrolledimages/" + i + '/' + sidetext + '/' + j + '/bin.bin'
                        best_match_string = "./enrolledimages/" + i + '/' + sidetext + '/' + j
        key, matches = compare_binfiles('./tempeye/bin.bin', best_match_path, rois1cache, keycache1cache)
        self.load_matches()
        
        print(best_match_string)
        print(main_keypoints)
        print(matches)
        print(str(best_match_value * 100) + "% match")
        
        matchjsontable = {}
        with open(best_match_string + '/info.json', 'r') as openfile:
            # Reading from json file
            matchjsontable = json.load(openfile)
            print(matchjsontable)
        
        
        positivity = ""
        if(matchjsontable["groundtruth"] == Path(image_curr).name):
            print("Positive")
            positivity = "Positive"
        else:
            print("Negative")
            positivity = "Negative"
            
        info = ""
        info += str(best_match_path) + "\n"
        info += str(matches) + "\n"
        info += str(str(best_match_value * 100) + "% match") + "\n"
        
        #Cache JSON for Recognition
        if not os.path.isdir("./recognition_cache"):
            os.mkdir("./recognition_cache")
        
        #Takes from info.json of best match
        recognition_cache = {
            "eyebeingrecognized" : Path(image_curr).name,
            "groundtruth" : matchjsontable["groundtruth"],     
            "groundtruth_eyename" : matchjsontable["eyename"],
            "side" : matchjsontable["side"],
            "best_match_string" : best_match_string,
            "main_keypoints" : main_keypoints,
            "matches" : matches,
            "best_match_value" : best_match_value
        }
        
        with open("./recognition_cache/" + Path(image_curr).name + '.json', 'w') as openfile:
            # Reading from json file
            y = json.dumps(recognition_cache, indent=4)
            openfile.write(y)
            
        #info += str(positivity) + "\n"
        self.textEdit_4.setText(info)
        self.label_40.setPixmap(QtGui.QPixmap("./enrolledimages/" + i + '/' + sidetext + '/' + j + '/equalized histogram iris region.jpg'))
    def thisenroll(self):
        global eyechanged
        side = 0 if self.radioButton.isChecked() else 1 if self.radioButton_2.isChecked() else 0
        if self.radioButton_7.isChecked():
            print("Enrolling")
            eyename = self.textEdit.toPlainText() if not self.textEdit.toPlainText() == None else "Anonymous"
            if curr_image != "" and eyechanged:
                enrolleye(curr_image, eyename, side)
                path = ""
                with open('path.txt') as f:
                    path = f.readlines()
                self.label_39.setPixmap(QtGui.QPixmap(str(path[0]) + "/complete.jpg"))
                eyechanged = False
            else:
                print("Invalid Image or Eye Already Enrolled")
        elif self.radioButton_8.isChecked():
            print(curr_image)
            self.recognize(str(curr_image), side)
    def masstest(self):
        if self.radioButton_7.isChecked():
            #Mass Enroll
            path = "./CASIA1/**"
            globity = glob.glob(path, recursive=True)
            for path in globity:
                if path.endswith('jpg'):
                    print(path, path.split('\\')[-1].split('_')[-3], path.split('\\')[-1].split('_')[-2])
                    if path.split('\\')[-1].split('_')[-2] == '1' or path.split('\\')[-1].split('_')[-2] == '2':
                        enrolleye(path, '1eye_' + path.split('\\')[-1].split('_')[-3], 0 if path.split('\\')[-1].split('_')[-2] == '1' else 1 if path.split('\\')[-1].split('_')[-2] == '2' else 0)
        elif self.radioButton_8.isChecked():
            path = "./CASIA1/**"
            globity = glob.glob(path, recursive=True)
            
            for path in globity:
                if path.endswith('jpg'):
                    #print(path)
                    path = path.split('/')[-1].split('\\')
                    side = int(path[-1].split('_')[-2]) - 1
                    print(path)
                    print(side)
                    pathsub = ""
                    for i in path:
                        pathsub += '/' + i 
                    #print(path)
                    mainpath = pathlib.Path(__file__).parent.resolve()
                    #print(mainpath)
                    global curr_image
                    curr_image = str(mainpath) + str(pathsub)
                    print(curr_image)
                    global eyechanged
                    eyechanged = True
                    if curr_image != "":
                        self.recognize(str(curr_image), side)
                    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Process"))
        self.radioButton.setText(_translate("MainWindow", "Left Eye"))
        self.radioButton_2.setText(_translate("MainWindow", "Right Eye"))
        self.label_2.setText(_translate("MainWindow", "Eye Label"))
        self.radioButton_7.setText(_translate("MainWindow", "Enroll"))
        self.radioButton_8.setText(_translate("MainWindow", "Recognize"))
        self.label_4.setText(_translate("MainWindow", "Side"))
        self.label_5.setText(_translate("MainWindow", "Mode"))
        self.label_14.setText(_translate("MainWindow", "Current Eye"))
        self.label_15.setText(_translate("MainWindow", "Best Match"))
        self.textEdit_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_28.setText(_translate("MainWindow", "Left Match"))
        self.label_30.setText(_translate("MainWindow", "Bottom Match"))
        self.label_34.setText(_translate("MainWindow", "Right Match"))
        self.label_35.setText(_translate("MainWindow", "Complete Match"))
        self.label_36.setText(_translate("MainWindow", "Command"))
        self.textEdit_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_16.setText(_translate("MainWindow", "Match Info"))
        self.textEdit_5.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_17.setText(_translate("MainWindow", "Eye Info"))
        self.label_18.setText(_translate("MainWindow", "Settings"))
        self.pushButton_2.setText(_translate("MainWindow", "Mass Process"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
