import cv2
import sys
# from cftracker.csrdcf import CSRDCF
# from cftracker.MRScale_estimator import MREstimator as CSRDCF
from cftracker.csrdcf_mr import CSRDCF_MR as CSRDCF
from cftracker.config import csrdcf_config


def create_tracker():
    tracker=CSRDCF(csrdcf_config.CSRDCFConfig())
    return tracker

start_index = 29

def main():
    # tracker = CSRDCF()
    tracker = create_tracker()
    video = cv2.VideoCapture('/home/shawn/scripts_output_tmp/WorldChampionJet06.avi')
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    i = 0
    while True:
        ok, frame = video.read()
        if i < start_index:
            i += 1
            comp = 8 - len(str(i))
            strl = "0" * comp + str(i)
            # cv2.imwrite(
            #     "/home/shawn/scripts_output_tmp/RRR_detected_results/exp43/worldChampion_tracked/frame_%s.jpg" % strl,
            #     frame)
            continue

        if ok is None or frame is None:
            break
        else:
            i += 1
            #frame = cv2.resize(frame, (960, 540))

        if i == start_index+1:
            # box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            box = (278, 166, 697, 284)  # 29 frame
            # box = (379, 168, 474, 285)  # 150 frame
            tracker.init(frame, box)
        bbox = tracker.update(frame)

        if True:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
            cv2.putText(frame, "Tracked", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        comp = 8-len(str(i))
        strl = "0"*comp+str(i)
        print("This is frame "+strl)
        # cv2.imwrite("/home/shawn/scripts_output_tmp/huiguiren_MR_square/frame_%s.jpg"%strl, frame)
        #
        cv2.imshow("", frame)
        key = cv2.waitKey(1) & 0xFF
        pass

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
