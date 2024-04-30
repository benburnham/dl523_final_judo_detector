import numpy as np

def find_closest_box(boxes, frame_width, frame_height):
    center_x = frame_width // 2
    center_y = frame_height // 2

    box_centers_x = (boxes[:, 0, 0] + boxes[:, 1, 0]) // 2
    box_centers_y = (boxes[:, 0, 1] + boxes[:, 1, 1]) // 2

    distances = np.sqrt((box_centers_x - center_x)**2 + (box_centers_y - center_y)**2)
    closest_indices = np.argsort(distances)

    closest_box1 = boxes[closest_indices[0]]
    closest_box2 = boxes[closest_indices[1]] if len(boxes) > 1 else None

    return closest_box1, closest_box2

def find_closest_box_slow(boxes, frame_width, frame_height):
    
    center_x = frame_width // 2
    center_y = frame_height // 2
    min_distance = float('inf')
    
    closest_box1 = None
    closest_box2 = None

    for box in boxes:
        box = [round(i, 2) for i in box.tolist()]
        int_box = [int(x) for x in box]
        # Calculate the center of the box
        box_center_x = (int_box[0] + int_box[2]) // 2
        box_center_y = (int_box[1] + int_box[3]) // 2

        # Calculate the distance from the center of the image to the center of the box
        # np.sqrt(((x1+x2)/2-640)**2 + ((y1+y2)/2-360)**2)
        distance = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)

        # Update the closest box if necessary
        if distance < min_distance:
            min_distance = distance
            closest_box2 = closest_box1
            closest_box1 = box

    return closest_box1, closest_box2

# Example usage
# image_shape = (480, 640)  # Example image shape (height, width)
# boxes = [((100, 100), (200, 200)), ((300, 300), (400, 400)), ((200, 200), (300, 300))]  # Example list of boxes
# closest_box = find_closest_box(boxes, image_shape)
# print("Closest Box:", closest_box)
