import numpy as np
from enum import Enum
from typing import Dict, List
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


class LmType(Enum):
    FaceLm = (468, 3)
    HandLm = (21, 3)
    PoseLm = (33, 2)


def lm_mp(lm_arr: np.ndarray):
    holistic = mp.solutions.holistic.Holistic()
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    holistic_res = holistic.process(test_img)
    holistic_res.face_landmarks = lm_from_np(
        lm_arr[:1404], lm_type=LmType.FaceLm)
    holistic_res.right_hand_landmarks = lm_from_np(
        lm_arr[1404:1467], lm_type=LmType.HandLm)
    holistic_res.left_hand_landmarks = lm_from_np(
        lm_arr[1467:1530], lm_type=LmType.HandLm)
    holistic_res.pose_landmarks = lm_from_np(
        lm_arr[1530:1596], lm_type=LmType.PoseLm)
    return holistic_res


def lm_from_np(arr: np.ndarray, lm_type=LmType):
    normList = landmark_pb2.NormalizedLandmarkList()
    res_list: List[List[float]] = arr.reshape(lm_type.value).tolist()
    if lm_type != LmType.PoseLm:
        for res in res_list:
            lm = normList.landmark.add()
            lm.x = res[0]
            lm.y = res[1]
            lm.z = res[2]
    else:
        for res in res_list:
            lm = normList.landmark.add()
            lm.x = res[0]
            lm.y = res[1]
            lm.z = 0

    return normList
