import cv2
import numpy as np
import tensorflow as tf


#This program classifies handwritten digits in 
#a Region of Interest from the screen. 
#The classification is done with a conv neural
#network previously trained on the MNIST dataset.

#Written by Carlos Gutierrez (cgutierrez.eng@gmail.com)

#loads model trained on the MNIST database
load_model = tf.keras.models.load_model


#coordinates of the ROI
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 800

#threshold level 
THRES_LEVEL = 200
#minimum accuracy of the classification
ACCURACY_LEVEL = 0.8
#minimum area from the contour
#to try classification
MIN_AREA = 300



#classifies an image and returns classification and 
#percentage of confidence
def classifier(frame):
    frame_resized = cv2.resize(frame, (28, 28))
    frame_resized_inv = cv2.bitwise_not(frame_resized)

    frame_array = np.array(frame_resized_inv.reshape(1, 28, 28, 1))

    frame_array = frame_array.astype("float32") /255

    # Load the model from the SavedModel directory
    model = load_model('my_model_digits')  

    classification = model.predict(frame_array, verbose = 0)
    digit_index = classification[0].argmax()
    confidence = classification[0][digit_index]

    return (digit_index, confidence)


def add_white_padding(image):
    """
    Pads an image with white pixels to make its width and height the same.

    Args:
    image: A numpy array representing the image.

    Returns:
    A numpy array representing the padded image.
    """
    # Get the image shape
    image_height, image_width = image.shape[:2]
    # Find the maximum dimension
    max_dim = max(image_height, image_width)

    #creates white square 
    white_image = np.full((max_dim, max_dim), 255, dtype=np.uint8)

    #padding is added to the sides
    if image_height > image_width:
        total_padding = max_dim - image_width
        side_padding = total_padding // 2

        x_offset = side_padding
        y_offset = 0

        white_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

        return (white_image)
    #passing is added in top and bottom
    if image_width > image_height:

        total_padding = max_dim - image_width
        upper_padding = total_padding // 2

        x_offset = 0
        y_offset = upper_padding

        white_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

        return (white_image)
    
    elif image_height == image_width:
        return (white_image)




def main():
    cam = cv2.VideoCapture(0)


    while True:
        #captures frame
        ret, frame = cam.read()
            
        # clone the frame
        frame_copy = frame.copy()
        
        # Draw ROI Rectangle on frame copy
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
        
        # Grab the ROI from the frame
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]
        
        # Apply grayscale and blur to ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        
        #applies binary threshold
        ret , thresholded = cv2.threshold(gray, THRES_LEVEL, 255, cv2.THRESH_BINARY)

        # Define a kernel for erotion
        kernel = np.ones((3, 3), np.uint8)
    
        # Erodes the image to thicken the black portion
        thresholded = cv2.erode(thresholded, kernel, iterations=3)
            
        #gets contours in the ROI 
        contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        #This list stores internal contours found
        internal_contours = []
    
        #Populates internal_contours list
        for i in range(len(contours)):
        # Check if the contour is an inner contour
            if hierarchy[0][i][3] != -1:
                internal_contours.append(contours[i])

        #if at least one internal contour was found, then gets the 
        #largest one and draw it
        if len(internal_contours) > 0:
            
            #goes through every countour found to atttemp a 
            #classification
            for contour in internal_contours:
            
                #only attempts to classify contours
                #that are of a certain area
                if cv2.contourArea(contour) > MIN_AREA:
                
                    cv2.drawContours(frame_copy, [contour + (roi_right, roi_top)], -1, (255, 0, 0),5)
                    # Get the bounding box coordinates
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw the bounding box around the contour on the original image
                    cv2.rectangle(frame_copy, (x+roi_right, y+roi_top), (x+w+roi_right, y+h+roi_top), (0, 255, 0), 2)
            
                    #focused_roi is the area of the contour detected inside roi
                    focused_roi = thresholded[y:y+h, x:x+w]
                    #cv2.imshow("focused",focused_roi)


                    focused_roi_padded = add_white_padding(focused_roi)
                
                    #attempts to classify the area of focus inside thresholded
                    results = classifier(focused_roi_padded)

                    #if confidence percentage returned by the clasfiffier is higher than
                    #ACCURACY_LEVEL then display label and percentange in green. Else in red
            
                    accuracy = round(results[1] * 100, 0)
                    classification = str(results[0])

                    label = f"{classification}-{accuracy}%"
                    label_position = (x+roi_right, y + roi_top - 10)
            
                    if results[1] > ACCURACY_LEVEL:
                       
                        cv2.putText(frame_copy, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Blue text with thickness of 2

                    else:
                
                        cv2.putText(frame_copy, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Blue text with thickness of 2
        
                
                    #if confidence percentage returned by the clasfiffier is higher than
                    #ACCURACY_LEVEL then display ROI frame in green
                    if results[1] > ACCURACY_LEVEL:
                        # Draw ROI Rectangle on frame copy
                        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,255,0), 5)
                

        cv2.imshow("Live",frame_copy)
        cv2.imshow("Number",thresholded)
        
        
        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            # Save the image to disk
            #cv2.imwrite('image.jpg', focused_roi)
            break

    # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

