import csv
import numpy as np
import tensorflow as tf
import cv2
import time
import os

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

        #  print("Elapsed Time:", end_time-start_time)                        ##### THIS LINE IS COMMENT BY ME.

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
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


model_path = '/Users/sahilkirti/Desktop/Final Mini Project 2/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.7
videos_dir = '/Users/sahilkirti/Desktop/Final Mini Project 2/all_videos/videos(120-130)'
output_dir = '/Users/sahilkirti/Desktop/Final Mini Project 2/output_videos(100-110)'

odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.7
# Create a list to store the spatio-temporal interest points
all_interest_points = []
prev_frame = None
frame_number = 0
video=0
for video_file in os.listdir(videos_dir):
    if video_file.endswith(".avi"):
        video_path = os.path.join(videos_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_file = os.path.join(output_dir, f"{video_name}_descriptors_with_labels.csv")

        # Open video capture object
        video_capture = cv2.VideoCapture(video_path)
        
        total_valid_cntr = 0
        descriptors = []

        # Iterate over frames in the video
        count=0
        while count<25:
            print('frame number',count)
            ret, frame = video_capture.read()
            if not ret:
                break
            
            boxes, scores, classes, num = odapi.processFrame(frame)

            frame_boxes = []
            for i in range(len(boxes)):
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                    frame_boxes.append(box)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None and prev_frame.shape == gray_frame.shape:
                optical_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                harris_corners = cv2.cornerHarris(gray_frame, 2, 3, 0.04)
                harris_corners = cv2.dilate(harris_corners, None)
                interest_points_indices = np.argwhere(harris_corners > 0.01 * harris_corners.max())
                
                for point in interest_points_indices:
                    frame_number, x, y = frame_number, point[1], point[0]
                    for i in range(len(frame_boxes)):
                        box = frame_boxes[i]
                        x1, x2, y1, y2 = box[1], box[3], box[0], box[2]
                        flag = 0
                        if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                            interest_point = (frame_number, x, y)
                            all_interest_points.append(interest_point)
                            frame[y][x] = [0, 0, 255]
                            flag = 1
                        if flag == 1:
                            break

                descriptors = []
                for frame_number, x, y in all_interest_points:
                    patch = gray_frame[y-16:y+16, x-16:x+16]
                    if patch.shape != (32, 32):
                        continue
                    hog_features = extract_hog_features(patch)
                    flow_patch = optical_flow[y-16:y+16, x-16:x+16]
                    if flow_patch.shape != (32, 32, 2):
                        continue
                    hof_features = extract_hof_features(flow_patch)
                    descriptor = np.concatenate((hog_features, hof_features))
                    descriptors.append(descriptor.tolist() + [video_name])

                total_valid_cntr += len(descriptors)

            prev_frame = gray_frame
            count+=1
            if(count<22):
              break

        # Save the descriptors to a CSV file
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f"HOG_feature_{i+1}" for i in range(hog_features.shape[0])] + \
                     [f"HOF_feature_{i+1}" for i in range(hof_features.shape[0])] + \
                     ["Label"]
            writer.writerow(header)  # Write the header
            writer.writerows(descriptors) # Write the descriptors for each frame
        print("succesful video",video)
        video+=1
        # Release video capture
        video_capture.release()
