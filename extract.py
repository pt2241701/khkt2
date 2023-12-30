from ultralytics import YOLO
import cv2
import os
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
save_dir_label = r'D:\a\label'
save_dir_image = r'D:\a\image'
class_name = "d"
label = ["a", "b", "c", "o", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "p", "q", "r", "s", "m", "t", "u", "v", "w", "x", "y", "yes", "no", "me", "you", "hello", "i_love_you", "eat", "thank_you", "little", "sorry", "drink", "want", "space", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
counter = 1
while True:
    _, img = cap.read()
    cv2.resize(img, (640,640))
    predictions = model.predict(source=img, show=True)

    for result in predictions:
        boxe = result.boxes.cpu().numpy()     
        xywh = boxe.xywhn
        for xywhn in xywh:
            print(xywhn)
            # Save image
            image_filename = os.path.join(save_dir_image, f'{counter}_{class_name}.jpg')
            cv2.imwrite(image_filename, img)
            # Save label
            label_filename = os.path.join(save_dir_label, f'{counter}_{class_name}.txt')
            with open(label_filename, 'a') as label_file:
                label_file.write(f'{label.index(class_name)} {(xywhn[0])} {(xywhn[1])} {(xywhn[2])} {(xywhn[3])}')
            # Increment counter
            counter += 1
    # cv2.imshow('YOLO V8 Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()