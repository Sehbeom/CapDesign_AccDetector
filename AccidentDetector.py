from inspect import currentframe
from turtle import TPen
# import boto3
import cv2
import re
import math

# 영상 및  txt 파일명 변수
global local_file
global track_info
global obj_file
global img_file
global lines

# N Frames 단위 수행 관련 변수
global blockSize
global isFinished
global isClipped
global ClippedFrame

# 속도 및 가속도, Bounding Box Width 변화량 데이터 저장 변수
global velocity
global change_of_velocity
global bboxWidth
global Frame_Objects

# 각 알고리즘 별 Threshold 설정 변수 
global threshold_Peak_V
global threshold_Peak_W
global threshold_PAPR_V
global threshold_PAPR_W
global threshold_MinMaxScaler_V
global threshold_MinMaxScaler_W

# ==== 변수 초기화 ====
# N Frames 단위 수행 시, blockSize = N, 1초 = 30frames
lines = []
blockSize = 300
isFinished = False
isClipped=False
ClippedFrame=0

threshold_Peak_V = 42
threshold_Peak_W = 42

threshold_PAPR_V = 140
threshold_PAPR_W = 42

threshold_MinMaxScaler_V = 0.095
threshold_MinMaxScaler_W = 0.095

# AWS S3 Upload 함수
def upload_clip() :
    s3 = boto3.client('s3')
    try:
        s3.upload_file(obj_file,BUCKET_NAME,obj_file,
        ExtraArgs={'ContentType' : 'video/mp4', 'ACL': 'public-read'}
        )
    except Exception as err:
        print("upload error",err)

# Clipping 함수
def clipping(AccidentFrame) :
    fourcc = cv2.VideoWriter_fourcc(* 'h264')

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
    print("cap settings finished")
    while(True):
        ret, frame = cap.read()
        if currentframe == int((endFrame-startFrame)/2) :
            cv2.imwrite(img_file,frame)
        
        if currentframe >= endFrame - startFrame:
            break

        if ret == False:
            print("retret")
            continue

        currentframe = currentframe + 1

        writer.write(frame) 

        if cv2.waitKey(1) & 0xFF ==27:
            break

    writer.release()
    cv2.destroyAllWindows()

# Tracking 결과 데이터 N Frames 단위로 변수에 저장
def getObjectsDataNFrames(seq) :
    global blockSize
    global length
    global isFinished
    global isClipped
    global ClippedFrame
    
    if seq==1:
        startFrame=1
        endFrame=blockSize
    else:
        startFrame = (seq-1)*(int(blockSize/2))
        endFrame = startFrame+blockSize

    Frame_Number = 0
    object_number = 0
    for line in lines:
        if line == "\n":
            continue
        elif len(line) < 6:
            Frame_Number = int(re.findall("\d+",str(line))[0])
            if Frame_Number == length:
                isFinished=True
                break
            
            if (isClipped==True) and ((Frame_Number - ClippedFrame)>blockSize*5):
                isClipped=False

            objects = {}
            continue
        
        if (Frame_Number>=startFrame) and (Frame_Number<=endFrame):
            object_number = float(re.findall("\d+\.\d+",str(line))[0])
            objects[int(object_number)] = tuple(re.findall("\d+\.\d+",str(line))[1:5])
            Frame_Objects[Frame_Number] = objects
        
        elif Frame_Number>endFrame:
            break

# 가속도 데이터 분석
def Calculate_Distance_Traveled(ObjData) :
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
                     # 다음 Frame에도 그 객체가 존재한다면
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

# Bounding Box Width 변화량 데이터 분석
def Calculate_Width(ObjData) :
    prev_value=float(0)
    isFirst=0
    target_velocity = {}
    for One_Frame in ObjData: # 각 Frame 별로 분리됨 -> index는 Frame count // one_Frame = Frame Number
        for obj in ObjData[One_Frame] : # 한 프레임 내 객체별로 분리됨 -> obj는 Tracking ID
            first_key = next(iter(ObjData))
            for One_Frame in ObjData:
                if obj in ObjData[One_Frame]:
                    if isFirst==0:
                        prev_value=float(ObjData[One_Frame][obj][2])
                        isFirst=isFirst+1
                    elif isFirst==1:
                     # 다음 Frame에도 그 객체가 존재한다면
                        target_velocity[(One_Frame)] = abs(float(ObjData[One_Frame][obj][2]) - float(prev_value))
                        prev_value=float(ObjData[One_Frame][obj][2])
                    
            bboxWidth[obj] = target_velocity
            target_velocity = {}
            prev_value=float(0)
            isFirst=0

# PAPR 적용 및 분석
def Get_PAPR_Info(InputData):
    target_change = {} 
    prev_value=float(0)
    cnt = 0

    maximum_PAPR = 0 # obj 들의 PAPR 값들중 가장큰 값을 담는 변수
    maximum_Frame = 0 # 가장큰 PAPR 값이 위치한 Frame

    for Object in InputData: # 각 객체별
        obj_sum = 0
        mean = 0
        if len(InputData[Object]) < 15:
            continue
        first_key = next(iter(InputData[Object]))
        for frame in InputData[Object]: # 시간별
            cnt = cnt + 1 # PAPR or min-max 평균 계산시 나눠주는 전체 Frame 수
            if frame==first_key:
                prev_value=float(InputData[Object][frame])
            else:
                target_change[frame] = abs(InputData[Object][frame] - prev_value)

                prev_value=InputData[Object][frame]
                obj_sum = obj_sum + target_change[frame] # PAPR 계산에서 평균을 계산하기 위한 합
        mean = float(obj_sum / cnt) # PAPR 계산시 사용되는 obj별 BBOX 가속도 평균

        for frame2 in target_change: # PAPR 계산식 + obj들의 PAPR들중 가장 큰 PAPR값 찾는 부분
            target_change[frame2] = pow(float(target_change[frame2]),2) / pow(mean,2)
            if target_change[frame2] > maximum_PAPR:
                maximum_PAPR = target_change[frame2]
                maximum_Frame = frame2

        target_change = {}
        prev_value=float(0)
        cnt = 0
    

    return (maximum_PAPR, maximum_Frame)

# 교통사고 판별 알고리즘 - PAPR
def isAccident_PAPR_NFrames(InputData,threshold) :
    global obj_file
    global img_file
    global isClipped
    global ClippedFrame
    maximum_PAPR, maximum_Frame = Get_PAPR_Info(InputData)
    AccidentFrame=0

    if (maximum_PAPR > threshold) and isClipped==False: #// PAPR 기반 Threshold의 기준
        AccidentFrame = maximum_Frame
        clipping(AccidentFrame)
        print(AccidentFrame)

        obj_file = 'result_acc/AccFrame'+str(AccidentFrame)+'.mp4'
        img_file = 'result_acc/AccFrame'+str(AccidentFrame)+'.jpg'

        isClipped=True
        ClippedFrame = AccidentFrame

    return AccidentFrame

def Get_MinMaxScaler_Info(InputData):
    target_change = {} 
    prev_value=float(0)
    cnt = 0
    min_mean = 1 # Normalized 평균의 최솟값을 담는 변수
    min_mean_Frame = 0
    for Object in InputData: # 각 객체별
        obj_sum = 0
        temporary_accidentFrame = 0
        minimum = 2 # min-max scaling에 사용
        maximum = 0 # obj들의 peak들중 가장큰 값을 담는 변수 or min-max scaling에 사용
        if len(InputData[Object]) < 15:
            continue
        first_key = next(iter(InputData[Object]))
        for frame in InputData[Object]: # 시간별
            cnt = cnt + 1 # PAPR or min-max 평균 계산시 나눠주는 전체 Frame 수
            if frame==first_key:
                prev_value=float(InputData[Object][frame])
            else:
                target_change[frame] = abs(InputData[Object][frame] - prev_value)
                if target_change[frame] < minimum: # min-max 계산부
                    minimum = target_change[frame]
                elif target_change[frame] > maximum:
                    maximum = target_change[frame]

                prev_value=InputData[Object][frame]
        
        for frame2 in target_change: # Normalized mean 계산
            target_change[frame2] = float((target_change[frame2] - minimum) / (maximum - minimum))
            obj_sum = target_change[frame2] + obj_sum
            if target_change[frame2] == 1:
                temporary_accidentFrame = frame2

        if min_mean > float(obj_sum / cnt): # Normalized mean의 최소 지점을 찾기
            min_mean = float(obj_sum / cnt)
            min_mean_Frame = temporary_accidentFrame

        target_change = {}
        prev_value=float(0)
        cnt = 0
    
    return (min_mean, min_mean_Frame)

# 교통사고 판별 알고리즘 - MinMaxScaler
def isAccident_MinMaxScaler_NFrames(InputData,threshold) :
    global obj_file
    global img_file
    global isClipped
    global ClippedFrame
    min_mean, min_mean_Frame = Get_MinMaxScaler_Info(InputData)

    AccidentFrame = 0

    if (min_mean < threshold)  and isClipped==False: # Normalized 평균 기반 Thrshold 기준
        AccidentFrame = min_mean_Frame
        clipping(AccidentFrame)
        print(AccidentFrame)
        obj_file = 'result_acc/AccFrame'+str(AccidentFrame)+'.mp4'
        img_file = 'result_acc/AccFrame'+str(AccidentFrame)+'.jpg'

        isClipped=True
        ClippedFrame = AccidentFrame
    return AccidentFrame


#=========== 단일 영상 동작 ==============

# 파일 이름 설정
local_file = 'accVideos/cutFrameTest2.avi'
track_info = 'accVideos/cutFrameTest2.txt'
obj_file = 'result_acc/cutFrameTestResult2.mp4'
img_file = 'result_acc/cutFrameTestResult2.jpg'

# 영상 Capture 시작 및 fps, 전체 Frames 수 받아오기
cap = cv2.VideoCapture(local_file)
fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# global 변수 초기화
velocity = {}
bboxWidth = {}
change_of_velocity = {}
Frame_Objects = {}

# 트랙킹 결과 정보 받아오기
with open(track_info,'r') as f:
        lines = f.readlines()

# ============= N Frames 단위 동작 ================
infoIndex = 1

while(isFinished==False):
    getObjectsDataNFrames(infoIndex)
    print(str(infoIndex)+"'s Sequence Start\n")

    infoIndex=infoIndex+1

    if isFinished==False:
        Calculate_Distance_Traveled(Frame_Objects)
        # AccFrame = isAccident_PAPR_NFrames(velocity, threshold_PAPR_V)
        AccFrame = isAccident_MinMaxScaler_NFrames(velocity, threshold_MinMaxScaler_V)

        Frame_Objects.clear()
        velocity.clear()
        change_of_velocity.clear()

cap.release()