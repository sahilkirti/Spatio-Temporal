import csv
import numpy as np
import tensorflow as tf
import cv2
import time
import os
from concurrent.futures import ThreadPoolExecutor

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()  # Change here
            with tf.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:  # Change here
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)  # Change here

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        #  print("Elapsed Time:", end_time-start_time)                      

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(box  -zes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def extract_hog_features(image, cell_size=(8, 8), block_size=(2, 2), nbins=4):
    win_sz=(image.shape[1] // cell_size[1] * cell_size[1],image.shape[0] // cell_size[0] * cell_size[0])

    hog = cv2.HOGDescriptor(_winSize=win_sz,
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)
    hog_features = hog.compute(image)
    return hog_features.flatten()

# Function to extract HOF features
def extract_hof_features(flow, cell_size=(8, 8), nbins=5):
    # Calculate magnitude and direction of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    # Quantize direction into nbins
    bin_size = 360 // nbins
    quantized_angle = (angle / bin_size).astype(np.int32) % nbins

    # Create HOF descriptor
    hof_features = []
    for y in range(0, flow.shape[0], cell_size[0]):
        for x in range(0, flow.shape[1], cell_size[1]):
            cell_magnitude = magnitude[y:y+cell_size[0], x:x+cell_size[1]]
            cell_angle = quantized_angle[y:y+cell_size[0], x:x+cell_size[1]]
            hist, _ = np.histogram(cell_angle, bins=nbins, weights=cell_magnitude, range=(0, nbins))
            hof_features.extend(hist)

    return np.array(hof_features)


# Open a video capture object
model_path = '/Users/sahilkirti/Desktop/Final Mini Project 2/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.7
video_path='/Users/sahilkirti/Desktop/Final Mini Project 2/all_videos/videos(80-90)/7606-2_700810.avi'
video_capture = cv2.VideoCapture(video_path)  # Replace 'input_video.mp4' with the actual video file path
video_name = os.path.basename(video_path)

total_valid_cntr=0
# Create a variable to store the previous frame and frame number
prev_frame = None
frame_number = 0

# Create a list to store the spatio-temporal interest points
all_interest_points = []

while True:
   
    # Read a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break
    print("frame number",frame_number)  
    # Increment frame number
    frame_number += 1
    boxes, scores, classes, num = odapi.processFrame(frame)

    frame_boxes = []
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            # print(box)
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            frame_boxes.append(box)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using the Lucas-Kanade method
    if prev_frame is not None:
        optical_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Perform Harris corner detection on the grayscale framed
        harris_corners = cv2.cornerHarris(gray_frame, 2, 3, 0.04)

        # Dilate the corner points to mark them
        harris_corners = cv2.dilate(harris_corners, None)
        
        # Define a threshold for optimal corner detection
        interest_points_indices = np.argwhere(harris_corners > 0.01 * harris_corners.max())
        
        # Store the interest points for this frame
        cntr=0
        cntr2=0
        for point in interest_points_indices:
          cntr+=1
          frame_number, x, y = frame_number, point[1], point[0]
          for i in range(len(frame_boxes)):
            box=frame_boxes[i]
            x1=box[1]
            x2=box[3]
            y1=box[0]
            y2=box[2]
            flag=0
            # print(x1,y1,x2,y2)
            if(x>=x1 and x<=x2 and y>=y1 and y<=y2):
                cntr2+=1
                interest_point = (frame_number, x, y)
                all_interest_points.append(interest_point)
                frame[y][x]=[0,0,255]
                flag=1
            if flag==1:
                break
        descriptors = []
        for frame_number, x, y in all_interest_points:
           # Get a patch around the interest point
           patch = gray_frame[y-16:y+16, x-16:x+16] 
           if patch.shape != (32, 32):
            continue 

           # Extract HOG features
           hog_features = extract_hog_features(patch)

           
           flow_patch = optical_flow[y-16:y+16, x-16:x+16]
           if flow_patch.shape != (32, 32, 2):
            continue
           hof_features = extract_hof_features(flow_patch)

           # Concatenate HOG and HOF features
        
           descriptor = np.concatenate((hog_features, hof_features))
          # descriptors.append(descriptor.tolist() + [video_name])
           descriptors.append(descriptor.tolist() + [os.path.splitext(video_name)[0]]) 


        total_valid_cntr +=cntr
         
 
  
      #  cv2.imshow('Spatio-Temporal Interest Points', frame)

        # Check for 'q' key press to exit
      #  if cv2.waitKey(1) & 0xFF == ord('q'):
      #      break

    # Update the previous frame for the next iteration
    prev_frame = gray_frame
    if(frame_number==25):
        break

# Release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
print(total_valid_cntr)

# Save the interest points to a CSV file
with open('7606-2_700810.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    header = [f"HOG_feature_{i+1}" for i in range(hog_features.shape[0])] + \
             [f"HOF_feature_{i+1}" for i in range(hof_features.shape[0])] + \
             ["Label"]
    writer.writerow(header)  # Write the header
    writer.writerows(descriptors) # Write the interest points for each frame
