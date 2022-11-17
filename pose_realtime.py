import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

Emergency_path = "C:/Users/SEONGKUK/pose_dcgan/Emergency/"

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return round(angle,2)






# cap = cv2.VideoCapture('C:/Users/SEONGKUK/pose_dcgan/video.mp4')
#cap = cv2.VideoCapture('C:/Users/SEONGKUK/pose_dcgan/choke.mp4')

cap = cv2.VideoCapture('C:/Users/SEONGKUK/pose_dcgan/test.mp4')

#cap = cv2.VideoCapture(0)

# Curl counter variables
left_counter = 0
right_counter =0
stage = None

num_choke=0
num_fall=0

pic_choke=0
pic_fall=0

count=0

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # 왼쪽 어깨、 팔꿈치、 손목 、엉덩이、 무릎 좌표 정의
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]



            # 오른쪽 어깨、 팔꿈치、 손목 、엉덩이、 무릎 좌표 정의
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]






            # 이웃한 부위의 각도 계산
            
            # sew: 어깨-팔꿈치-손목목
            sew_left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            sew_right_angle = calculate_angle(right_shoulder, right_elbow, left_wrist)


            # shk: 어깨-엉덩이-무릎
            shk_left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            shk_right_angle = calculate_angle(right_shoulder, right_hip, left_knee)


            # hkh: 엉덩이-무릎-발목
            hkh_left_angle = calculate_angle(left_hip, left_knee, left_heel)
            hkh_right_angle = calculate_angle(right_hip, right_knee, left_heel)




            # 어깨-팔꿈치-손목 각도값 화면 출력
            cv2.putText(image, str(sew_left_angle), 
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(sew_right_angle), 
                           tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )                                
            
            # # Curl counter logic
            # if sew_left_angle > 160:
            #     stage = "down"
            # if sew_left_angle < 30 and stage =='down':
            #     stage="up"
            #     left_counter +=1
            #     print("left : "+left_counter)

            # if sew_right_angle > 160:
            #     stage = "down"
            # if sew_right_angle < 30 and stage =='down':
            #     stage="up"
            #     right_counter +=1
            #     print("right : "+right_counter)




            # 어깨-엉덩이-무릎 각도값 화면 출력
            cv2.putText(image, str(shk_left_angle), 
                           tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(shk_right_angle), 
                           tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )                                
            
            # # Curl counter logic
            # if shk_left_angle > 160:
            #     stage = "down"
            # if shk_left_angle < 30 and stage =='down':
            #     stage="up"
            #     left_counter +=1
            #     print("left : "+left_counter)

            # if shk_right_angle > 160:
            #     stage = "down"
            # if shk_right_angle < 30 and stage =='down':
            #     stage="up"
            #     right_counter +=1
            #     print("right : "+right_counter)








            cv2.putText(image, str(hkh_left_angle), 
                           tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(hkh_right_angle), 
                           tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )                                
            
            # Curl counter logic

    

#elif (120<sew_left_angle and sew_left_angle<180) and (30<sew_right_angle and sew_right_angle<70):
#    stage = "FALL!!"


# 60<sew_right_angle<160   40<hkh_left_angle<150  30<hkh_right_angle<140


#elif(sew_right_angle>60 and sew_right_angle<160) and (40<hkh_left_angle and hkh_left_angle<150) and (30<hkh_right_angle and hkh_right_angle<140):
#    stage = "FALL!!"



            if (80<sew_left_angle and sew_left_angle<120) and (30<hkh_left_angle and hkh_left_angle<100):
                num_fall=num_fall+1
                if (num_fall>10): # choke 상태가 일정시간 유지될 경우 응급상황으로 간주

                    stage="FALL!!"
                    timestr = time.strftime("%Y%m%d_%H%M%S")
                    img_captured2 = cv2.imwrite(Emergency_path+timestr+'.png',frame)
                    pic_fall=pic_fall+1


                    print("\n\ngoogle_drive 실행 전")

                    if(pic_fall==4):

                        import google_drive
                        print("google_drive 실행 후")
                        num_fall=num_fall+1
                    
            

            elif (sew_left_angle<40 and sew_left_angle>10)and (sew_right_angle>10 and sew_right_angle<40) and abs(sew_left_angle - sew_right_angle)<15:
                num_choke=num_choke+1
                if (num_choke>20 and num_choke%5==0): # choke 상태가 일정시간 유지될 경우 응급상황으로 간주
                    stage="CHOKE!!"
                    timestr = time.strftime("%Y%m%d_%H%M%S")
                    img_captured = cv2.imwrite(Emergency_path+timestr+'.png',frame)
                    pic_choke=pic_choke+1
                    print("\n\ngoogle_drive2 실행 전")


                    if(pic_choke==4):
                        num_choke=num_choke+1
                    
                        import google_drive2
                        print("google_drive2 실행 후")



# 넘어졌다가 일어날 때 순간을 포착해냄
            # elif (120<sew_left_angle and sew_left_angle<180) and (30<sew_right_angle and sew_right_angle<70)<15:
            #     num_fall=num_fall+1
            #     if num_fall>30:
            #         stage="FALL!!"
            #         timestr = time.strftime("%Y%m%d_%H%M%S")
            #         img_captured2 = cv2.imwrite(Emergency_path+timestr+'.png',frame)
            #         count=count+1
            # #         # if(count>3):
            # #         #     import google_drive
            # #         #     break
                    

            # elif (sew_right_angle>60 and sew_right_angle<160) and (40<hkh_left_angle and hkh_left_angle<100) and (30<hkh_right_angle and hkh_right_angle<100)<15:
            #     num_fall=num_fall+1
            #     if num_fall>20:
            #         stage="FALL!!"
            #         timestr = time.strftime("%Y%m%d_%H%M%S")
            #         img_captured2 = cv2.imwrite(Emergency_path+timestr+'.png',frame)
            #         count=count+1


            else :
                stage = "FINE"


            
                       
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # # Rep data
        # cv2.putText(image, 'REPS', (15,12), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(left_counter), 
        #             (10,60), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        # cv2.putText(image, 'STAGE', (65,12), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    