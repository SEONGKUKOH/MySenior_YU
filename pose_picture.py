import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


fall_path = 'C:/Users/SEONGKUK/pose_dcgan/fall_picture/'
sleeping_path = 'C:/Users/SEONGKUK/pose_dcgan/sleeping_picture/'
watching_path = 'C:/Users/SEONGKUK/pose_dcgan/watching_picture/'
emergency_path = 'C:/Users/SEONGKUK/pose_dcgan/emergency_picture/'


fall_pose_path = 'C:/Users/SEONGKUK/pose_dcgan/fall_picture_pose/'
sleeping_pose_path = 'C:/Users/SEONGKUK/pose_dcgan/sleeping_picture_pose/'
watching_pose_path = 'C:/Users/SEONGKUK/pose_dcgan/watching_picture_pose/'
emergency_pose_path = 'C:/Users/SEONGKUK/pose_dcgan/emergency_picture_pose/'

csv_path = 'C:/Users/SEONGKUK/pose_dcgan/csv/'


fall_list = os.listdir(fall_path)
sleeping_list = os.listdir(sleeping_path)
watching_list = os.listdir(watching_path)
emergency_list = os.listdir(emergency_path)


IMAGE_FALL = fall_list.copy()
IMAGE_SLEEP = sleeping_list.copy()
IMAGE_WATCH = watching_list.copy()
IMAGE_EMERGENCY = emergency_list.copy()


arr_fall = np.empty((0,8),int)
arr_sleeping = np.empty((0,8),int)
arr_watching = np.empty((0,8),int)
arr_emergency = np.empty((0,8),int)





def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return round(angle,2)







# For static images:
#IMAGE_FILES = ['images3.png','images2.jpg']
BG_COLOR = (192, 192, 192) # gray




with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:



  # 잠자는 자세  
  for idx, file in enumerate(sleeping_list):
    situation = "sleep"
    image = cv2.imread(sleeping_path + file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      print("not result")
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width},'
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )


    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    

    landmarks = results.pose_landmarks.landmark
        
    # Get coordinates
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]





    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]





    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]








    # Calculate angle
    sew_left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    sew_right_angle = calculate_angle(right_shoulder, right_elbow, left_wrist)


    shk_left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    shk_right_angle = calculate_angle(right_shoulder, right_hip, left_knee)


    hkh_left_angle = calculate_angle(left_hip, left_knee, left_heel)
    hkh_right_angle = calculate_angle(right_hip, right_knee, left_heel)





    # Visualize angle
    annotated_image = cv2.putText(annotated_image, str(sew_left_angle), 
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

    annotated_image = cv2.putText(annotated_image, str(sew_right_angle), 
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )                                
    






    annotated_image = cv2.putText(image, str(shk_left_angle), 
                          tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )

    annotated_image = cv2.putText(image, str(shk_right_angle), 
                          tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )                                





    annotated_image = cv2.putText(image, str(hkh_left_angle), 
                        tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

    annotated_image = cv2.putText(image, str(hkh_right_angle), 
                        tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )                                
          




    # 특정 부위에서 얻은 각도 좌표값을 넘파이로 저장한다
    arr_sleeping = np.append(arr_sleeping,np.array([[int(idx+1),situation,sew_left_angle,sew_right_angle,shk_left_angle,shk_right_angle,hkh_left_angle,hkh_right_angle]]),axis=0)




    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
         )
    cv2.imwrite(sleeping_pose_path + str(idx+1) + '.png', annotated_image)






  # 넘어지는 자세
  for idx, file in enumerate(fall_list):
    situation = "fall"
    image = cv2.imread(fall_path + file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      print("not result")
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width},'
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )


    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    

    landmarks = results.pose_landmarks.landmark
        
    # Get coordinates
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]





    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]





    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]








    # Calculate angle
    sew_left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    sew_right_angle = calculate_angle(right_shoulder, right_elbow, left_wrist)


    shk_left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    shk_right_angle = calculate_angle(right_shoulder, right_hip, left_knee)


    hkh_left_angle = calculate_angle(left_hip, left_knee, left_heel)
    hkh_right_angle = calculate_angle(right_hip, right_knee, left_heel)





    # Visualize angle
    annotated_image = cv2.putText(annotated_image, str(sew_left_angle), 
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

    annotated_image = cv2.putText(annotated_image, str(sew_right_angle), 
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )                                
    






    annotated_image = cv2.putText(image, str(shk_left_angle), 
                          tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )

    annotated_image = cv2.putText(image, str(shk_right_angle), 
                          tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )                                





    annotated_image = cv2.putText(image, str(hkh_left_angle), 
                        tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

    annotated_image = cv2.putText(image, str(hkh_right_angle), 
                        tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )                                
          




    # 특정 부위에서 얻은 각도 좌표값을 넘파이로 저장한다
    arr_fall = np.append(arr_fall,np.array([[int(idx+1),situation,sew_left_angle,sew_right_angle,shk_left_angle,shk_right_angle,hkh_left_angle,hkh_right_angle]]),axis=0)




    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
         )
    cv2.imwrite(fall_pose_path + str(idx+1) + '.png', annotated_image)






# 응급상황
  for idx, file in enumerate(emergency_list):
    situation = "emergency"
    image = cv2.imread(emergency_path + file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      print("not result")
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width},'
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )


    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    

    landmarks = results.pose_landmarks.landmark
        
    # Get coordinates
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]





    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]





    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]








    # Calculate angle
    sew_left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    sew_right_angle = calculate_angle(right_shoulder, right_elbow, left_wrist)


    shk_left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    shk_right_angle = calculate_angle(right_shoulder, right_hip, left_knee)


    hkh_left_angle = calculate_angle(left_hip, left_knee, left_heel)
    hkh_right_angle = calculate_angle(right_hip, right_knee, left_heel)





    # Visualize angle
    annotated_image = cv2.putText(annotated_image, str(sew_left_angle), 
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

    annotated_image = cv2.putText(annotated_image, str(sew_right_angle), 
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )                                
    






    annotated_image = cv2.putText(image, str(shk_left_angle), 
                          tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )

    annotated_image = cv2.putText(image, str(shk_right_angle), 
                          tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )                                





    annotated_image = cv2.putText(image, str(hkh_left_angle), 
                        tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

    annotated_image = cv2.putText(image, str(hkh_right_angle), 
                        tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )                                
          




    # 특정 부위에서 얻은 각도 좌표값을 넘파이로 저장한다
    arr_emergency = np.append(arr_emergency,np.array([[int(idx+1),situation,sew_left_angle,sew_right_angle,shk_left_angle,shk_right_angle,hkh_left_angle,hkh_right_angle]]),axis=0)




    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
         )
    cv2.imwrite(emergency_path + str(idx+1) + '.png', annotated_image)








  # tv보는 자세  
  for idx, file in enumerate(watching_list):
    situation = "watching"
    image = cv2.imread(watching_path + file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      print("not result")
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width},'
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )


    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    

    landmarks = results.pose_landmarks.landmark
        
    # Get coordinates
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]





    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]





    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]








    # Calculate angle
    sew_left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    sew_right_angle = calculate_angle(right_shoulder, right_elbow, left_wrist)


    shk_left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    shk_right_angle = calculate_angle(right_shoulder, right_hip, left_knee)


    hkh_left_angle = calculate_angle(left_hip, left_knee, left_heel)
    hkh_right_angle = calculate_angle(right_hip, right_knee, left_heel)





    # Visualize angle
    annotated_image = cv2.putText(annotated_image, str(sew_left_angle), 
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

    annotated_image = cv2.putText(annotated_image, str(sew_right_angle), 
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )                                
    






    annotated_image = cv2.putText(image, str(shk_left_angle), 
                          tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )

    annotated_image = cv2.putText(image, str(shk_right_angle), 
                          tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )                                





    annotated_image = cv2.putText(image, str(hkh_left_angle), 
                        tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

    annotated_image = cv2.putText(image, str(hkh_right_angle), 
                        tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )                                
          




    # 특정 부위에서 얻은 각도 좌표값을 넘파이로 저장한다
    arr_watching = np.append(arr_watching,np.array([[int(idx+1),situation,sew_left_angle,sew_right_angle,shk_left_angle,shk_right_angle,hkh_left_angle,hkh_right_angle]]),axis=0)




    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
         )
    cv2.imwrite(watching_pose_path + str(idx+1) + '.png', annotated_image)





























print("\n\n\n")


arr_all = np.concatenate((arr_fall,arr_sleeping,arr_watching,arr_emergency),axis=0)


df_sleeping = pd.DataFrame(arr_sleeping, columns=['name','situation','sew_left_angle','sew_right_angle','shk_left_angle','shk_right_angle','hkh_left_angle','hkh_right_angle'])
df_fall = pd.DataFrame(arr_fall, columns=['name','situation','sew_left_angle','sew_right_angle','shk_left_angle','shk_right_angle','hkh_left_angle','hkh_right_angle'])
df_watching = pd.DataFrame(arr_fall, columns=['name','situation','sew_left_angle','sew_right_angle','shk_left_angle','shk_right_angle','hkh_left_angle','hkh_right_angle'])
df_emergency = pd.DataFrame(arr_fall, columns=['name','situation','sew_left_angle','sew_right_angle','shk_left_angle','shk_right_angle','hkh_left_angle','hkh_right_angle'])
df_all = pd.DataFrame(arr_all, columns=['name','situation','sew_left_angle','sew_right_angle','shk_left_angle','shk_right_angle','hkh_left_angle','hkh_right_angle'])



print(df_sleeping)

print("\n\n\n\n")

print(df_fall)

print("\n\n\n\n")

print(df_watching)

print("\n\n\n\n")

print(df_emergency)

print("\n\n\n\n")

print(df_all)

pd.DataFrame(df_sleeping).to_csv(csv_path+'sleeping.csv')
pd.DataFrame(df_fall).to_csv(csv_path+'fall.csv')
pd.DataFrame(df_fall).to_csv(csv_path+'watching.csv')
pd.DataFrame(df_fall).to_csv(csv_path+'emergency.csv')
pd.DataFrame(df_all).to_csv(csv_path+'all.csv')




    # Plot pose world landmarks.(meidapipe 포즈를 x,y,z 차트화해서 보여주는 함수)
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)



    # print("result.pose_landmarks")
    # print(results.pose_landmarks)




#    print("result.pose_world_landmarks")
#    print(results.pose_world_landmarks)



