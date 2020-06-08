from threading import Thread
import time

import classifierAlexa
import classifier_pyqt5


def main():
    try:
        classifierAlexa_thread = Thread(target=classifierAlexa.app.run)
        classifierAlexa_thread.start()

        time.sleep(1)
        classifier_thread = Thread(target=classifier_pyqt5.startApp())
        classifier_thread.start()
    except Exception:
        print("Unknown exception occurred!")
        raise


if __name__ == '__main__':
    main()
