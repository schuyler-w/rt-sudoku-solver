import solver
import tensorflow as tf
import cv2


def main():
    model = tf.keras.models.load_model("./modelData.h5")
    cap = cv2.VideoCapture(0)
    
    
## if __name__ == "__main__":
##    main()