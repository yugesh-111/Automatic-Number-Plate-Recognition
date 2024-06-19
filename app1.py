import cv2
import streamlit as st
import numpy as np
import os
import datetime
harcascade = "haarcascade_russian_plate_number.xml"
min_area = 500 
count = 0
output_folder='scanned_plates'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def detect_plate_image(image):
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    files_in_output_folder = len([name for name in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, name))])
    count = files_in_output_folder

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Save the detected number plates
            count += 1
            cropped_plate = image[y:y + h, x:x + w]

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{output_folder}/detected_plate_{count}_{timestamp}.jpg"
            cv2.imwrite(filename, cropped_plate)

    st.image(image, channels="BGR")


def main():
    st.title('🔍 Number Plate Detector 🚗')
    activities = ['Live Stream 🎥', 'Upload Image 📤']
    choice = st.sidebar.selectbox("Select Activity", activities, index=1)  # Set default to 'Upload Image'

    if choice == 'Live Stream 🎥':
        st.subheader("Live Stream 🔴")

        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # width
        cap.set(4, 480)  # height
        plate_cascade = cv2.CascadeClassifier(harcascade)

        while True:
            success, img = cap.read()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

            for (x, y, w, h) in plates:
                area = w * h
                if area > min_area:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                    
                    files_in_output_folder = len([name for name in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, name))])
                    count = files_in_output_folder

                    count += 1
                    cropped_plate = img[y:y + h, x:x + w]

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{output_folder}/detected_plate_{count}_{timestamp}.jpg"
                    cv2.imwrite(filename, cropped_plate)
            stframe.image(img, channels="BGR")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif choice == 'Upload Image 📤':
        st.subheader("Upload Image 🔵")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(image, -1)
            detect_plate_image(img)


if __name__ == '__main__':
    main()
