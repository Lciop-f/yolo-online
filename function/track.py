from dateutil.parser import parser
from function.tracker.byte_tracker import BYTETracker
import cv2
import argparse

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class Mytracker:
    def __init__(self,args):
        self.args = args
        self.tracker = BYTETracker(self.args)
    def draw_online_targets(self,image, online_targets):
        for target in online_targets:
            tlwh = target.tlwh
            tid = target.track_id
            tlbr = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
            color = (0, 255, 0)
            cv2.rectangle(image, (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3])), color, 2)
            cv2.putText(image, f"ID: {tid}", (int(tlbr[0]), int(tlbr[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image

    def track(self, dets, info_imgs, img_size,image):
        online_targets = self.tracker.update(dets, info_imgs, img_size)
        return self.draw_online_targets(image, online_targets)




