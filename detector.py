import cv2
import numpy as np
import time

mobilenet_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (255, 0, 255)]
yolo_color = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

weightsPath = "models/yolov4.weights"
configPath = "models/yolov4.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
layerOutput = net.getUnconnectedOutLayersNames()

netSSD = cv2.dnn.readNet("models/ssd_mobilenet_v1.pb", "models/ssd_mobilenet_v1.pbtxt")
# netSSD.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# netSSD.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# netSSD.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
netSSD.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

cap = cv2.VideoCapture('video.mp4')
frame_w, frame_h = cap.get(3), cap.get(4)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(frame_w)*2, int(frame_h)))

frame = 0
mobilenet_latency = 0
yolo_latency = 0

while cap.isOpened():
  ret, image = cap.read()
  if ret != True:
    break

  frame += 1
  imageYOLO = image.copy()
  start_time = time.time()
  blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (192, 192), [0, 0, 0], swapRB=False, crop=False)
  net.setInput(blob)
  layer_outputs = net.forward(layerOutput)

  classIds = []
  confidences = []
  boxes = []
  confThreshold = 0.4
  nmsThreshold = 0.3
  
  for output in layer_outputs:
    for detection in output:
      scores = detection[5:]
      classId = np.argmax(scores)
      confidence = scores[classId]
      c_x = int(detection[0] * frame_w)
      c_y = int(detection[1] * frame_h)
      w = int(detection[2] * frame_w)
      h = int(detection[3] * frame_h)
      x = int(c_x - w / 2)
      y = int(c_y - h / 2)
      classIds.append(classId)
      confidences.append(float(confidence))
      boxes.append([x, y, w, h])
  
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

  for i in idxs:    
    box = boxes[i]
    cv2.rectangle(imageYOLO, (box[0],box[1]), (box[0]+box[2], box[1]+box[3]), yolo_color[classIds[i]], 2)

  end_time = time.time() - start_time
  end_time *= 1000
  yolo_latency += end_time
  cv2.putText(imageYOLO, f"YOLO ({end_time} ms)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

  # MobileNet SSD
  imageSSD = image.copy()
  start_time = time.time()
  
  netSSD.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
  netSSDOut = netSSD.forward()

  for detection in netSSDOut[0,0,:,:]:
    score = float(detection[2])
    if score > confThreshold:
        left = detection[3] * frame_w
        top = detection[4] * frame_h
        right = detection[5] * frame_w
        bottom = detection[6] * frame_h
        cv2.rectangle(imageSSD, (int(left), int(top)), (int(right), int(bottom)), mobilenet_color[int(detection[1]) - 1], thickness=2)


  end_time = time.time() - start_time
  end_time *= 1000
  mobilenet_latency += end_time
  cv2.putText(imageSSD, f"MobileNet ({end_time} ms)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
  # cv2.imwrite(f"res/{time.time()}.jpg", image)
  res = np.concatenate((imageSSD, imageYOLO), axis=1)
  cv2.imshow('frame', np.concatenate((imageSSD, imageYOLO), axis=1))
  out.write(res)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
out.release() 

print("===========================")
# print("DNN BACKEND: OPENCV")
print("DNN BACKEND: IE")
# print("DNN TARGET: CPU")
print("DNN TARGET: MYRIAD")
print("===========================")

print(f"Mobilenet avg latency: {mobilenet_latency/frame} ms")
print(f"YOLO avg latency: {yolo_latency/frame} ms")
# cv2.destroyAllWindows()
