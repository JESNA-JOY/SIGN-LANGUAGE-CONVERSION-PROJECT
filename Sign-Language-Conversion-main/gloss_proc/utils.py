import mediapipe as mp
import numpy as np
import pandas as pd
import os
import torch
import itertools
from typing import List, Tuple
from functools import reduce

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Drawing from mediapipe holistic docs : )

GData = Tuple[np.ndarray, List[str]]


def draw_landmarks(image: np.ndarray, res) -> np.ndarray:
    mp_drawing.draw_landmarks(
        image,
        res.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        res.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        res.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        res.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    return image


def set_gloss_path(gloss: str, gloss_dir: str) -> str:
    data_path = os.path.join(os.getcwd(), gloss_dir)
    req_dir = os.path.join(data_path, gloss)
    if not os.path.exists(req_dir):
        os.mkdir(req_dir)
    return req_dir


def get_all_gloss(gloss_dir: str) -> List[str]:
    if os.path.exists(gloss_dir):
        return list(filter(lambda fname: os.path.isdir(os.path.join(gloss_dir, fname)), os.listdir(gloss_dir)))
    else:
        return []


def gdata_dir(gloss_dir: str) -> str:
    data_path = os.path.join(
        os.getcwd(), gloss_dir)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    return data_path


def get_all_vid(gloss_dir: str, gloss: str) -> List[str]:
    gpath = os.path.join(gdata_dir(gloss_dir), gloss)
    return list(filter(lambda fname: os.path.isfile(
        fname), map(lambda f: os.path.join(gpath, f), os.listdir(gpath))))


def gdata_count(gloss_dir: str, gloss: str) -> int:
    try:
        return len(get_all_vid(gloss_dir, gloss))
    except:
        return 0


def get_vid_data(vid_dir: str) -> np.ndarray:
    return pd.read_csv(vid_dir, header=None).to_numpy()


def _get_gdata(gloss_dir: str, gloss: str) -> GData:
    gd: np.ndarray = np.asarray([get_vid_data(vid)
                                for vid in get_all_vid(gloss_dir, gloss)])
    return (gd, list(itertools.repeat(gloss, gd.shape[0])))


def _get_all_gdata(seq_size: int, gloss_dir: str) -> GData:
    acc: GData = (np.empty((0, seq_size, 1596)), [])
    for gloss in get_all_gloss(gloss_dir):
        data, label = _get_gdata(gloss_dir, gloss)
        acc = (np.append(acc[0], data, axis=0), acc[1]+label)
    return acc


