#!/usr/bin/env python
# coding: utf-8

# In[157]:


import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import glob
import pandas as pd
import mediapipe as mp
from constants import LIPS_POSITIONS, FACE_OVAL,HAND_POSITIONS,HAND_CONNECTIONS
from google.protobuf.json_format import MessageToDict
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import NormalizedLandmark
import h5py

DEFAULT_LEFT_HAND = np.load("defaultLeftHand.npy")
DEFAULT_RIGHT_HAND = np.load("defaultRightHand.npy")
DEFAULT_FACE_OVAL = np.load("defaultFaceOval.npy")
DEFAULT_FACE_LIPS = np.load("defaultFaceLips.npy")


# In[2]:


class Drawing:
    def __init__(self):
        self.mpFace = mp.solutions.face_mesh
        self.mpDrawHands = mp.solutions.drawing_utils # Initializing drawing object for hands
        self.mpDrawFace = mp.solutions.drawing_utils # Initializing drawing object for Face
        self.mp_drawing_styles =mp.solutions.drawing_styles
        self.mp_drawing_face = self.mpDrawFace.DrawingSpec(color=(0,0,200),thickness=0,circle_radius=1) #Initializing drawing specifications for face
        self.mp_drawing_hands = self.mpDrawHands.DrawingSpec(color=(255,0,0),thickness=0,circle_radius=1) #Initializing drawing specifications for hand
        self.mpHands = mp.solutions.hands
    def drawLandmarks(self,img,faceLandmarks,handLandmarks,img_size=(700,720)):
        img=img.copy()
        colors={"Right":(100,100,100),"Left":(0,0,255)}
        if faceLandmarks is not None:
            for key in faceLandmarks:
                for var in faceLandmarks[key]:
                    cv2.circle(img, (int(var[0]*img_size[0]),int(var[1]*img_size[1])), 1, (0, 0, 255), -1)
        for key in handLandmarks:
            points={}
            for i,var in enumerate(handLandmarks[key]):
                point = (int(var[0]*img_size[0]),int(var[1]*img_size[1]))
                cv2.circle(img, point, 3, colors[key], -1)
                points[i]=point
            for conn in HAND_CONNECTIONS:
                cv2.line(img, points[conn], points[HAND_CONNECTIONS[conn]], (216, 223, 230), 2)
        return img
        


# In[3]:


class HandLandmarkExtractor:
    def getHandLandmarks(self,hands,scale=False,img_size=(700,720)):
        for key in hands:
            list_hand_positions=[]
           # print(type(resultsFace.multi_face_landmarks[0]))

            for cord in HAND_POSITIONS:
                x1,y1,z1=self.__getCoordinates(hands[key],cord,scale,img_size)

                list_hand_positions.append((x1,y1,z1))
            hands[key]= np.array(list_hand_positions)    
        return hands    
    def __getCoordinates(self,landmarks,index,scale,img_size): 
        x=landmarks.landmark[index].x
        y=landmarks.landmark[index].y
        z=landmarks.landmark[index].z
        if scale:
            x=x*img_size[0]
            y=y*img_size[1]
        return x,y,z  

class FaceLandmarkExtractor:
    def __getLipsLandmarks(self,resultsFace,scale=False,img_size=(700,720)):
        list_lips_positions=[]
        if resultsFace.multi_face_landmarks:
            landmarkovi=resultsFace.multi_face_landmarks[0]

            for cord in LIPS_POSITIONS:
                x1,y1,z1=self.__getCoordinates(landmarkovi,cord[0],scale,img_size)
                x2,y2,z2=self.__getCoordinates(landmarkovi,cord[1],scale,img_size)

                avg_x=float((x1+x2)/2)
                avg_y=float((y1+y2)/2)

                list_lips_positions.append((avg_x,avg_y,z1))
        return np.array(list_lips_positions)
 
    def __getOvalFaceLandmarks(self,resultsFace,scale=False,img_size=(700,720)):
        list_face_positions=[]
       # print(type(resultsFace.multi_face_landmarks[0]))
        if resultsFace.multi_face_landmarks:
            landmarkovi=resultsFace.multi_face_landmarks[0]

            for cord in FACE_OVAL:
                x1,y1,z1=self.__getCoordinates(landmarkovi,cord,scale,img_size)

                list_face_positions.append((x1,y1,z1))
        return np.array(list_face_positions)
    
    def getFaceLandmarks(self,resultsFace,scale=False,img_size=(700,720)):
        face_landmarks={}
        face_landmarks["Lips"]=self.__getLipsLandmarks(resultsFace,scale,img_size)
        face_landmarks["Face"]=self.__getOvalFaceLandmarks(resultsFace,scale,img_size)
        return face_landmarks
    
    def __getCoordinates(self,landmarks,index,scale,img_size): 
        x=landmarks.landmark[index].x
        y=landmarks.landmark[index].y
        z=landmarks.landmark[index].z
        if scale:
            x=x*img_size[0]
            y=y*img_size[1]
        return x,y,z  


# In[4]:


class LandmarkExtractor:
    def __init__(self):
        self.mpHands = mp.solutions.hands # Load mediapipe hands module
        self.mpFace = mp.solutions.face_mesh
        self.hands = self.mpHands.Hands( # Initialize hands model
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False)
        
         # Load mediapipe face module
        self.faces = self.mpFace.FaceMesh( # Initialize Face model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False)
        self.handLandmarkExtractor=HandLandmarkExtractor()
        self.faceLandmarkExtractor=FaceLandmarkExtractor()

    def findHands(self,img):
        hands={}
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Transform to RGB
        results = self.hands.process(imgRGB) # Feeding image through Hands model
        if results.multi_handedness!=None:
            for i,hand in enumerate(results.multi_handedness):
                if hand.classification[0].label == "Left":
                    handType="Right"
                else:
                    handType="Left"
                hands[handType]=results.multi_hand_landmarks[i]


        return hands # Returning values from model prediction
    
    def findFace(self, img):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Transform image to RGB
        results = self.faces.process(imgRGB) # Feeding image through Face model
        return results
        
    def getFaceLandmarks(self,resultsFace,scale=False,img_size=(700,720)):
        return self.faceLandmarkExtractor.getFaceLandmarks(resultsFace,scale,img_size)
    def getHandLandmarks(self,resultsHand,scale=False,img_size=(700,720)):
        return self.handLandmarkExtractor.getHandLandmarks(resultsHand,scale,img_size)


# In[130]:


class VideoLoader:
    def __init__(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.landmark_extractor=LandmarkExtractor()
        self.drawing = Drawing()

    def __processFrame(self,frame):
        resultsFace=self.landmark_extractor.findFace(frame) #using function defined above to detect facial landmarks in a frame (findFace)
        faceLandmarks=self.landmark_extractor.getFaceLandmarks(resultsFace)
        
        resultsHands=self.landmark_extractor.findHands(frame)
        handLandmarks=self.landmark_extractor.getHandLandmarks(resultsHands)
        
        return faceLandmarks,handLandmarks
    def exportFeaturesToVideo(self,frames,features,output_path):
        outf = cv2.VideoWriter(output_path,self.fourcc, 15,(700,720))

        faceLandmarks={}
        handLandmarks={}

        for i,frame in enumerate(frames):
            faceLandmarks["Lips"]=features['faceLips'][i]
            faceLandmarks["Face"]=features['faceOval'][i]
            handLandmarks["Left"]=features['handLeft'][i]
            handLandmarks["Right"]=features['handRight'][i]
            outf.write(self.drawing.drawLandmarks(frame.copy(),faceLandmarks,handLandmarks))
            #out.write(self.drawing.drawLandmarks(frames[i].copy(),faceLandmarks,handLandmarks)) #drawing landmarks on frames by using function defined above (drawLadmarks)
        outf.release()

    def loadVideo(self,path,output_path=None):
        
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = fps/15
        if output_path is not None:
            out = cv2.VideoWriter(output_path,self.fourcc, 15,(700,720))

        frame_index = 0
        next_frame_to_use = 0.0

        frames=[]  
        while(True):
            ret, frame = cap.read() #reading frames
            if not ret:
                break
            if ret: #if frame exist ret=True, otherwise False
                if frame_index >= round(next_frame_to_use):
                    frame=frame[:, 300:1000,:] #cropping image, retainig all 3 rgb channels
                    frames.append(frame)
                    
                    if output_path is not None:
                        faceLandmarks,handLandmarks = self.__processFrame(frame)
                        out.write(self.drawing.drawLandmarks(frame.copy(),faceLandmarks,handLandmarks)) #drawing landmarks on frames by using function defined above (drawLadmarks)
        
                    next_frame_to_use += step
            frame_index += 1
        if output_path is not None:
            out.release() #close writing stream
        return frames


# In[225]:


class HandState:
    def __init__(self, side, default):
        self.side = side  # "Left" or "Right"
        self.default = default
        self.last = None
        self.missing_count = 0

    def update(self, handLandmarks, features):
        key = f"hand{self.side}"
        if self.side in handLandmarks:
            current = handLandmarks[self.side]
            features[key].append(current)

            if 0 < self.missing_count <= 25 and self.last is not None:
                for i in range(1, self.missing_count + 1):
                    features[key][-(i + 1)] = self.last
            self.missing_count = 0
            self.last = current
        else:
            features[key].append( self.default)
            self.missing_count += 1
class FaceState:
    def __init__(self, defaultLips,defaultOval):
        self.defaultOval = defaultOval
        self.defaultLips = defaultLips

        self.lastLips = None
        self.lastOval = None
        self.missing_count = 0

    def update(self, faceLandmarks, features):
        if faceLandmarks["Face"].size!=0 or faceLandmarks["Lips"].size!=0:
            features["faceLips"].append(faceLandmarks["Lips"])
            features["faceOval"].append(faceLandmarks["Face"])

            if 0 < self.missing_count <= 25 and self.lastOval is not None and self.lastLips is not None:
                for i in range(1, self.missing_count + 1):
                    features["faceLips"][-(i + 1)] = self.lastLips
                    features["faceOval"][-(i + 1)] = self.lastOval

            self.missing_count = 0
            self.lastOval = faceLandmarks["Face"]
            self.lastLips = faceLandmarks["Lips"]

        else:
            features["faceLips"].append(self.defaultLips)
            features["faceOval"].append(self.defaultOval)
            self.missing_count += 1
class FeatureExtraction:
    def __init__(self):
        self.landmark_extractor=LandmarkExtractor()
        self.video_loader=VideoLoader()
        self.dataframe=[]
    def __processFrame(self,frame):
        resultsFace=self.landmark_extractor.findFace(frame) #using function defined above to detect facial landmarks in a frame (findFace)
        faceLandmarks=self.landmark_extractor.getFaceLandmarks(resultsFace)
        
        resultsHands=self.landmark_extractor.findHands(frame)
        handLandmarks=self.landmark_extractor.getHandLandmarks(resultsHands)
        
        return faceLandmarks,handLandmarks
    
    def extractFromVideo(self,path,output_path=None):
        left_hand = HandState("Left", DEFAULT_LEFT_HAND)
        right_hand = HandState("Right", DEFAULT_RIGHT_HAND)
        face = FaceState(DEFAULT_FACE_LIPS,DEFAULT_FACE_OVAL)
        frames = self.video_loader.loadVideo(path)
        N = len(frames)
        features={'handLeft':[],'handRight':[],
                  'faceLips':[],'faceOval':[]}
        for r,frame in enumerate(frames):
            faceLandmarks,handLandmarks = self.__processFrame(frame)
            face.update(faceLandmarks,features)
            left_hand.update(handLandmarks, features)
            right_hand.update(handLandmarks, features)                
                    
        if output_path is not None:
            self.video_loader.exportFeaturesToVideo(frames,features,output_path)
        features={'handLeft':np.array(features['handLeft']),'handRight':np.array(features['handRight']),
                  'faceLips':np.array(features['faceLips']),'faceOval':np.array(features['faceOval'])}
        return features
    
    def saveFeatures(self, path,features):
        try:
            file_name=path.split("\\")[-1].split("-rgb_front")[0][:-2]

            with h5py.File(f"landmarks/{file_name}.h5", "w") as f:
                f.create_dataset("handLeft", data=features['handLeft'])
                f.create_dataset("handRight", data=features['handRight'])
                f.create_dataset("faceOval", data=features['faceOval'])
                f.create_dataset("faceLips", data=features['faceLips'])
            self.dataframe.append({"file_name": file_name,"landmarks":f"{file_name}.h5"})
        except Exception as e:
            print(e)
            print(path)
            features = self.extractFromVideo(path)
            self.saveFeatures(path,features)
    def getFeatures(self,folder_path):
        for path in glob.glob(f'{folder_path}/*.mp4'):

            features = self.extractFromVideo(path)
            self.saveFeatures(path,features)
        
        df = pd.DataFrame(self.dataframe)
        df.to_csv("AslLens-dataset.csv")


# In[226]:


featureExtraction = FeatureExtraction()


# In[227]:


featureExtraction.getFeatures("../../ASLens - test data 1")

