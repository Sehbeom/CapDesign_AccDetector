import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import boto3
import cv2
import re
import os
import math
import requests
from datetime import datetime

Ui_Form = uic.loadUiType("Capstone.ui")[0]

BUCKET_NAME = "capdesign-ict"

reqHeader = {"Content-type" : "application/json",
            "Authorization" : "key=SECRET INFO"}


global local_file
global track_info
global obj_file
global img_file

global velocity
global bboxWidth

global bboxHeight
global Frame_Objects
global objects

global threshold_PAPR
global threshold_Peak
global threshold_Normalized

threshold_Peak = 42
threshold_PAPR = 700
threshold_Normalized = 0.095

ACCESS_KEY = 'SECRET INFO'
SECRET_KEY = 'SECRET INFO'
SESSION_TOKEN = '...'

s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='ap-northeast-2'
)

def upload_clip(obj_file) :
    try:
        s3.upload_file(obj_file,BUCKET_NAME,obj_file,
        ExtraArgs={'ContentType' : 'video/mp4'}
        )
    except Exception as err:
        print("upload error",err)

def clipping(AccidentFrame,cap,obj_file,img_file) :
    fourcc = cv2.VideoWriter_fourcc(* 'h264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if AccidentFrame - 2*fps < 0 :
        startFrame = 0
    else :
        startFrame = AccidentFrame - 2*fps
    if AccidentFrame + 2*fps > length :
        endFrame = length
    else :
        endFrame = AccidentFrame + 2*fps
    currentframe = 0

    print(fps,width,height)

    writer = cv2.VideoWriter(obj_file,fourcc,fps,(width,height))

    cap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
    while(True):
        ret, frame = cap.read()

        if currentframe == int((endFrame-startFrame)/2) :
            cv2.imwrite(img_file,frame)
        
        if currentframe >= endFrame - startFrame:
            break

        if ret == False:
            continue

        currentframe = currentframe + 1

        writer.write(frame) 

        if cv2.waitKey(1) & 0xFF ==27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

def getObjectsData() :
    with open(track_info,'r') as f:
        lines = f.readlines()
        Frame_Number = 0
        object_number = 0
        for line in lines:
            if line == "\n":
                continue
            elif len(line) < 6:
                Frame_Number = int(re.findall("\d+",str(line))[0])
                objects = {}
                continue
            object_number = float(re.findall("\d+\.\d+",str(line))[0])
            objects[int(object_number)] = tuple(re.findall("\d+\.\d+",str(line))[1:5])
            Frame_Objects[Frame_Number] = objects

def Calculate_Distance_Traveled(ObjData,velocity) :
    distance_traveled = 0
    x_difference = 0
    y_difference = 0
    prev_x=float(0)
    prev_y=float(0)
    isFirst=0
    target_velocity = {}
    for One_Frame in ObjData: # 각 Frame 별로 분리됨 -> index는 Frame count // one_Frame = Frame Number
        for obj in ObjData[One_Frame] : # 한 프레임 내 객체별로 분리됨 -> obj는 Tracking ID
            for One_Frame in ObjData:
                if obj in ObjData[One_Frame]:
                    if isFirst==0:
                        prev_x=float(ObjData[One_Frame][obj][0])
                        prev_y=float(ObjData[One_Frame][obj][1])
                        isFirst=isFirst+1
                    elif isFirst==1:
                        x_difference = float(ObjData[One_Frame][obj][0]) - float(prev_x)
                        y_difference = float(ObjData[One_Frame][obj][1]) - float(prev_y)
                        distance_traveled = math.sqrt((x_difference ** 2) + (y_difference ** 2))
                        target_velocity[(One_Frame)] = distance_traveled
                        prev_x=float(ObjData[One_Frame][obj][0])
                        prev_y=float(ObjData[One_Frame][obj][1])
                    
            velocity[obj] = target_velocity
            target_velocity = {}
            prev_x=float(0)
            prev_y=float(0)
            isFirst=0

    return(velocity)

def Calculate_Change_Of_Distance_Traveled_Peak(VelocityData,cap,threshold,Acc_Count,obj_file,img_file,result) :
    target_change = {} 
    prev_value=float(0)
    cnt = 0
    AccidentFrame = 0
    maximum = 0 # obj들의 peak들중 가장큰 값을 담는 변수 or min-max scaling에 사용
    maxFrame = 0 # 가장큰 Peak가 위치한 Frame
    for Object in VelocityData: # 각 객체별
        if len(VelocityData[Object]) < 15:
            continue
        first_key = next(iter(VelocityData[Object]))
        for frame in VelocityData[Object]: # 시간별
            cnt = cnt + 1 # PAPR or min-max 평균 계산시 나눠주는 전체 Frame 수
            if frame==first_key:
                prev_value=float(VelocityData[Object][frame])
            else:
                target_change[frame] = abs(VelocityData[Object][frame] - prev_value)

                if maximum < target_change[frame]: #// obj들의 peak들중 가장 큰 peak를 찾는 부분
                    maximum = target_change[frame]
                    maxFrame = frame
                prev_value=VelocityData[Object][frame]
                
        target_change = {}
        prev_value=float(0)
        cnt = 0
    if maximum > threshold:  #// peak기반 threshold의 기준
        AccidentFrame = maxFrame
        clipping(AccidentFrame,cap,obj_file,img_file)
        upload_clip(obj_file)
        reqBody = {
            "to": "CLIENT TOKEN",
            "notification": {
            "title": "새로운 교통사고 영상이 업로드 되었습니다.",
            "body": obj_file,
            "click_action" : "http://localhost:3000/notice/" + obj_file
            }
        }
        response = requests.post("https://fcm.googleapis.com/fcm/send", 
            headers=reqHeader,
            json=reqBody)
        Acc_Count = Acc_Count + 1
        result = result + "Acc\n\n"
    else:
        result = result + "No Acc\n\n"

    return (AccidentFrame,Acc_Count,result)

def Calculate_Change_Of_Distance_Traveled_PAPR(VelocityData,cap,threshold,Acc_Count,obj_file,img_file,result) :
    target_change = {} 
    prev_value=float(0)
    cnt = 0
    AccidentFrame = 0
    maximum_PAPR = 0 # obj 들의 PAPR 값들중 가장큰 값을 담는 변수
    maxFrame_PAPR = 0 # 가장큰 PAPR 값이 위치한 Frame
    for Object in VelocityData: # 각 객체별
        obj_sum = 0
        mean = 0
        minimum = 2 # min-max scaling에 사용
        maximum = 0 # obj들의 peak들중 가장큰 값을 담는 변수 or min-max scaling에 사용
        if len(VelocityData[Object]) < 15:
            continue
        first_key = next(iter(VelocityData[Object]))
        for frame in VelocityData[Object]: # 시간별
            cnt = cnt + 1 # PAPR or min-max 평균 계산시 나눠주는 전체 Frame 수
            if frame==first_key:
                prev_value=float(VelocityData[Object][frame])
            else:
                target_change[frame] = abs(VelocityData[Object][frame] - prev_value)
                if target_change[frame] < minimum: # min-max 계산부
                    minimum = target_change[frame]
                elif target_change[frame] > maximum:
                    maximum = target_change[frame]

                prev_value=VelocityData[Object][frame]
                obj_sum = obj_sum + target_change[frame] # PAPR 계산에서 평균을 계산하기 위한 합
        mean = float(obj_sum / cnt) # PAPR 계산시 사용되는 obj별 BBOX 가속도 평균

        for frame2 in target_change: # PAPR 계산식 + obj들의 PAPR들중 가장 큰 PAPR값 찾는 부분
            target_change[frame2] = pow(float(target_change[frame2]),2) / pow(mean,2)
            if target_change[frame2] > maximum_PAPR:
                maximum_PAPR = target_change[frame2]
                maxFrame_PAPR = frame2

        target_change = {}
        prev_value=float(0)
        cnt = 0

    if maximum_PAPR > threshold: #// PAPR 기반 Threshold의 기준
        AccidentFrame = maxFrame_PAPR
        clipping(AccidentFrame,cap,obj_file,img_file)
        upload_clip(obj_file)
        reqBody = {
            "to": "CLIENT TOKEN",
            "notification": {
            "title": "새로운 교통사고 영상이 업로드 되었습니다.",
            "body": obj_file,
            "click_action" : "http://localhost:3000/notice/" + obj_file
            }
        }
        response = requests.post("https://fcm.googleapis.com/fcm/send", 
            headers=reqHeader,
            json=reqBody)
        Acc_Count = Acc_Count + 1
        result = result + "Acc\n\n"
    else:
        result = result + "No Acc\n\n"
        
    return (AccidentFrame,Acc_Count,result)

def Calculate_Change_Of_Distance_Traveled_Normalized(VelocityData,cap,threshold,Acc_Count,obj_file,img_file,result) :
    target_change = {} 
    prev_value=float(0)
    cnt = 0
    AccidentFrame = 0
    min_mean = 1 # Normalized 평균의 최솟값을 담는 변수
    
    for Object in VelocityData: # 각 객체별
        obj_sum = 0
        temporary_accidentFrame = 0
        minimum = 2 # min-max scaling에 사용
        maximum = 0 # obj들의 peak들중 가장큰 값을 담는 변수 or min-max scaling에 사용
        if len(VelocityData[Object]) < 15:
            continue
        first_key = next(iter(VelocityData[Object]))
        for frame in VelocityData[Object]: # 시간별
            cnt = cnt + 1 # PAPR or min-max 평균 계산시 나눠주는 전체 Frame 수
            if frame==first_key:
                prev_value=float(VelocityData[Object][frame])
            else:
                target_change[frame] = abs(VelocityData[Object][frame] - prev_value)
                if target_change[frame] < minimum: # min-max 계산부
                    minimum = target_change[frame]
                elif target_change[frame] > maximum:
                    maximum = target_change[frame]

                prev_value=VelocityData[Object][frame]
        
        for frame2 in target_change: # Normalized mean 계산
            target_change[frame2] = float((target_change[frame2] - minimum) / (maximum - minimum))
            obj_sum = target_change[frame2] + obj_sum
            if target_change[frame2] == 1:
                temporary_accidentFrame = frame2

        if min_mean > float(obj_sum / cnt): # Normalized mean의 최소 지점을 찾기
            min_mean = float(obj_sum / cnt)
            AccidentFrame = temporary_accidentFrame

        target_change = {}
        prev_value=float(0)
        cnt = 0

    if min_mean < threshold: # Normalized 평균 기반 Thrshold 기준
        clipping(AccidentFrame,cap,obj_file,img_file)
        upload_clip(obj_file)
        reqBody = {
            "to": "CLIENT TOKEN",
            "notification": {
            "title": "새로운 교통사고 영상이 업로드 되었습니다.",
            "body": obj_file,
            "click_action" : "http://localhost:3000/notice/" + obj_file
            }
        }
        response = requests.post("https://fcm.googleapis.com/fcm/send", 
            headers=reqHeader,
            json=reqBody)
        Acc_Count = Acc_Count + 1
        result = result + "Acc\n\n"
    else:
        result = result + "No Acc\n\n"

    return (AccidentFrame,Acc_Count,result)

class MyWindow(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Accident Handling System") # 폴더명 선택
        self.load_button.clicked.connect(self.folderopen)
        self.Play_Button.clicked.connect(self.Play)

    def folderopen(self):
        global foldername
        name = ""
        foldername = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder'))
        self.textEdit.setText(foldername)
        self.Notice.setText(" ")
        dirListing = os.listdir(foldername)
        dirListing.sort()
        for x in dirListing:
            if x[-3:] == "txt" or x[-5:] == "Store":
                continue
            name = name + x +"\n\n"
        self.name_list.setText(name)
        return foldername

    def Play(self):
        result = ""
        Acc_Count = 1
        dirListing = os.listdir(foldername)
        number_of_files = int((len(dirListing)/2))
        for i in range(1, number_of_files + 1):
            temp = Acc_Count
            local_file = foldername + '/video'+str(i)+'.avi'
            track_info = foldername + '/video'+str(i)+'.txt'
            now = datetime.now()
            current_time = now.strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
            obj_file = current_time+'.mp4'
            img_file = current_time+'.jpg'
            velocity = {}

            Frame_Objects = {}
            objects = {}

            cap = cv2.VideoCapture(local_file)

            with open(track_info,'r') as f:
                lines = f.readlines()
                Frame_Number = 0
                object_number = 0
                for line in lines:
                    if line == "\n":
                        continue
                    elif len(line) < 6:
                        Frame_Number = int(re.findall("\d+",str(line))[0])
                        objects = {}
                        continue
                    object_number = float(re.findall("\d+\.\d+",str(line))[0])
                    objects[int(object_number)] = tuple(re.findall("\d+\.\d+",str(line))[1:5])
                    Frame_Objects[Frame_Number] = objects

            velocity = Calculate_Distance_Traveled(Frame_Objects, velocity)
            #Frame_Number,Acc_Count,result = Calculate_Change_Of_Distance_Traveled_Peak(velocity,cap,threshold_Peak,Acc_Count,obj_file,img_file,result)
            Frame_Number,Acc_Count,result = Calculate_Change_Of_Distance_Traveled_PAPR(velocity,cap,threshold_PAPR,Acc_Count,obj_file,img_file,result)
            # Frame_Number,Acc_Count,result = Calculate_Change_Of_Distance_Traveled_Normalized(velocity,cap,threshold_Normalized,Acc_Count,obj_file,img_file,result)
            Frame_Objects.clear()
            velocity.clear()
            objects.clear()
            cap = 0
            self.progressBar.setValue(int((i/number_of_files)*100))
            if temp < Acc_Count:
                image = QtGui.QPixmap(img_file)
                QImage = image.scaled(QtCore.QSize(320,270))
                self.Accident_Image.setPixmap(QImage)
                self.Result.setText("사고가 발생했습니다. 홈페이지를 확인해주세요.")

        self.progressBar.setValue(100)
        self.result_list.setText(result)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MyWindow = MyWindow()
    MyWindow.show()
    app.exec_()