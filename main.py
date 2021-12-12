# Import the OpenCV and dlib libraries
import cv2
import dlib
import os
import numpy as np
import threading
import time

faceCascade = cv2.CascadeClassifier('frontalface_default.xml')

OUTPUT_SIZE_WIDTH = 800
OUTPUT_SIZE_HEIGHT = 600


def doRecognizePerson(faceNames, fid):
    time.sleep(2)
    faceNames[fid] = "Alumno " + str(fid)


def detectAndTrackMultipleFaces(clase):
    capture = cv2.VideoCapture('alumnos.mp4')

    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    cv2.startWindowThread()

    rectangleColor = (0, 128, 0)

    frameCounter = 0
    currentFaceID = 0

    # Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}

    count = 0


    try:
        while True:
            rc, fullSizeBaseImage = capture.read()

            if not rc: break
            baseImage = cv2.resize(fullSizeBaseImage, (450, 250))

            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('q'):
                break

            # Result image is the image we will show the user, which is a
            # combination of the original image from the webcam and the
            # overlayed rectangle for the largest face
            resultImage = baseImage.copy()

            # STEPS:
            # * Update all trackers and remove the ones that are not
            #   relevant anymore
            # * Every 10 frames:
            #       + Use face detection on the current frame and look
            #         for faces.
            #       + For each found face, check if centerpoint is within
            #         existing tracked box. If so, nothing to do
            #       + If centerpoint is NOT in existing tracked box, then
            #         we add a new tracker with a new face-id

            # Increase the framecounter
            frameCounter += 1

            fidsToDelete = []

            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[fid].update(baseImage)

                # If the tracking quality is good enough, we must delete
                # this tracker
                if trackingQuality < 7:
                    fidsToDelete.append(fid)

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop(fid, None)

            # Every 10 frames, we will have to determine which faces
            # are present in the frame
            if (frameCounter % 10) == 0:

                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                aux = baseImage.copy()

                faces = faceCascade.detectMultiScale(gray, 1.05, 6)

                for (_x, _y, _w, _h) in faces:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    # calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    # Variable holding information which faceid we
                    # matched with
                    matchedFid = None

                    # Now loop over all the trackers and check if the
                    # centerpoint of the face is within the box of a
                    # tracker
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        # calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        # check if the centerpoint of the face is within the
                        # rectangleof a tracker region. Also, the centerpoint
                        # of the tracker region must be within the region
                        # detected as a face. If both of these conditions hold
                        # we have a match
                        if ((t_x <= x_bar <= (t_x + t_w)) and
                                (t_y <= y_bar <= (t_y + t_h)) and
                                (x <= t_x_bar <= (x + w)) and
                                (y <= t_y_bar <= (y + h))):
                            matchedFid = fid




                    # If no matched fid, then we have to create a new tracker
                    if matchedFid is None:
                        print("Creating new tracker " + str(currentFaceID))

                        # Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage, dlib.rectangle(x, y, x + w, y + h ))

                        faceTrackers[currentFaceID] = tracker

                        # Start a new thread that is used to simulate
                        # face recognition. This is not yet implemented in this
                        # version :)
                        t = threading.Thread(target=doRecognizePerson, args=(faceNames, currentFaceID))
                        t.start()

                        # Increase the currentFaceID counter
                        currentFaceID += 1

            # Now loop over all the trackers we have and draw the rectangle
            # around the detected faces. If we 'know' the name for this person
            # (i.e. the recognition thread is finished), we print the name
            # of the person, otherwise the message indicating we are detecting
            # the name of the person

            for fid in faceTrackers.keys():
                tracked_position = faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)

                if fid in faceNames.keys():
                    cv2.putText(resultImage, faceNames[fid],
                                (int(t_x + t_w-10), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)

                    pathName = faceNames[fid]
                    if not os.path.exists(clase+'/'+pathName):
                        print('Carpeta personal creada: ', pathName)
                        os.makedirs(clase+'/'+pathName)

                    cv2.rectangle(baseImage, (t_x, t_y), (t_x + t_w, t_y + t_h), (255, 0, 0), 2)
                    rostro = aux[t_y:t_y + t_h, t_x:t_x + t_w]
                    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(clase+'/'+pathName + '/rostro_{}.jpg'.format(count), rostro)
                    count = count + 1

                    cv2.imshow('img', rostro)

                else:
                    cv2.putText(resultImage, "Detecting...",
                                (int(t_x + t_w - t_y/2), int(t_y)-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 255), 2)


            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))


            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)

    except KeyboardInterrupt as e:
        pass

    cv2.destroyAllWindows()



def entrenamiento(clase):
    dataPath = clase
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    faceData = []
    label = 0

    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imagenes de: ' + nameDir)

        for fileName in os.listdir(personPath):
            labels.append(label)
            faceData.append(cv2.imread(personPath + '/' + fileName, 0))
            image = cv2.imread(personPath + '/' + fileName, 0)

            cv2.imshow('image', image)
            cv2.waitKey(10)

        label = label + 1

    # Entrenamiento
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Entrenando...")
    face_recognizer.train(faceData, np.array(labels))

    # Almacenando el modelo obtenido
    face_recognizer.write(clase+'/modeloEntrenado.xml')
    print("Modelo almacenado...")



def reconocimientoFacial(clase):
    dataPath = clase
    imagePath = os.listdir(dataPath)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(clase+'/modeloEntrenado.xml')

    cap = cv2.VideoCapture('alumnos2.mp4')
    face_cascade = cv2.CascadeClassifier('frontalface_default.xml')



    while True:

        read, img = cap.read()

        if not read: break

        img = cv2.resize(img, (500, 300))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aux = gray.copy()

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            rostro = aux[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1] < 70:
                cv2.putText(img, '{}'.format(imagePath[result[0]]), (x, y-5), 2, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(img, 'Desconocido', (x, y - 5), 2, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        largeResult = cv2.resize(img, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

        cv2.imshow('img', largeResult)

        pressedKey = cv2.waitKey(2)
        if pressedKey == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    detectAndTrackMultipleFaces('1A')
    entrenamiento('1A')
    reconocimientoFacial('1A')
