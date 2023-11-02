import cv2
import os
import re
import numpy as np
import pandas as pd
import mediapipe as mp
import json
import gloss_proc.utils
from typing import List, NamedTuple, Union
from gloss_proc.utils import draw_landmarks, set_gloss_path, gdata_count, gdata_dir, get_all_gloss, _get_all_gdata, _get_gdata, GData

Landmark = NamedTuple("Landmark", [('x', float), (
    'y', float), (
        'z', float
)])
landmark = NamedTuple("landmark", [("landmark", List[Landmark])])
Landmarks = NamedTuple(
    'Landmarks', [
        ("right_hand_landmarks", landmark),
        ("left_hand_landmarks", landmark),
        ("pose_landmarks", landmark),
        ("face_landmarks", landmark),
    ])


def proc_landmarks(result_lmks: Landmarks) -> np.ndarray:
    # face_landmarks -> 468 x 3[x,y,z] -> 1404
    # hand_landmarks -> 21 x 2 [left,right] x 3[x,y,z] ->126
    # pose_landmarks -> 33 x 2[x,y, z discarded] -> 66
    # total  landmarks [face+hands+pose] -> 543 x (x,y,z in all landmarks except pose) ->1596

    face_lmks = result_lmks.face_landmarks if result_lmks.face_landmarks else None
    rt_hand_lmks = result_lmks.right_hand_landmarks if result_lmks.right_hand_landmarks else None
    lf_hand_lmks = result_lmks.left_hand_landmarks if result_lmks.left_hand_landmarks else None
    pose_lmks = result_lmks.pose_landmarks if result_lmks.pose_landmarks else None

    def map_lmk(landmarks: landmark, shape: int) -> np.ndarray:
        try:
            # Convert each landmark from [x:x_val,y:y_val,z:z_val] to [x_val,y_val,z_val] list and convert to array
            # and flatten the result getting a numpy vector
            #
            # eg : input => [
            #         { x:0.54, y:0.53, z.0.57},
            #         { x:0.48, y:0.54, z.0.61},
            #         { x:0.22, y:0.39, z.0.77}
            #        ]
            #
            #   output => [0.54,0.53,0.57,0.48,0.54,0.61,0.22,0.39,0.77]
            #
            return np.array(
                list(map(lambda l: [l.x, l.y, l.z], landmarks.landmark))).flatten()
        except Exception:
            # print("MapLmk Error : ", err)
            return np.zeros(shape)
            # all landmarks except pose landmark are passed to mapLmk to generate vector
            # pose landmark was not passed bcos z_val of pose_landmark is discarded
            # the result from map contains 3 numpy vectors  of shape [(1404,),(63,),(63,)]
            # pose landmarks are passed to map_pose function
            # The pose landmark vector is generated with shape (66,)
            # Each landmark is passed with shape bcos zero vector is generated  for each shape in the absence of landmark

    def map_pose(pose_lmks: landmark, shape: int) -> np.ndarray:
        try:
            return np.array(list(map(lambda l: [l.x, l.y], pose_lmks.landmark))).flatten()
        except Exception:
            # print("MapPose err : ", err)
            return np.zeros(shape)

    try:
        res = np.concatenate([
            map_lmk(face_lmks, 1404),
            map_lmk(rt_hand_lmks, 63),
            map_lmk(lf_hand_lmks, 63),
            map_pose(pose_lmks, 66),
        ])
    except Exception:
        res = np.zeros(1596)
        # print("Res error : ", err)
    return res


class GlossProcess():

    @staticmethod
    def default() -> "GlossProcess":
        return GlossProcess(get_all_gloss("gloss_data"))

    @staticmethod
    def load_checkpoint() -> "GlossProcess":
        try:
            with open('gproc_checkpnt.json', 'r') as file:
                chkpt = file.read()
            j = json.loads(chkpt)
            return GlossProcess(**j)
        except Exception as err:
            print("Error loading checkpoint : ", err)
            exit()
            # return GlossProcess.default()

    def __init__(self,
                 glosses: List[str] = [],
                 frame_count: int = 24,
                 vid_count: int = 20,
                 gloss_dir: str = "gloss_data",
                 append: bool = False,
                 skip: bool = False):

        # Gloss sanitization , minimum 1 gloss required
        if len(glosses) == 0 and len(glosses[0]) == 0:
            raise AttributeError("Invalid glosses")
        self.glosses = self._sanitize(glosses)
        # frame count default value : 24 @ 24fps ie 1 sec vid length
        self.frame_count = frame_count
        # total video count for each gloss  :  20
        self.vid_count = vid_count
        # directory to store gloss_data
        self.gloss_dir = gloss_dir
        # Generate gloss dir if it doesnot exist
        gdata_dir(gloss_dir)
        # Flag that specifies if data is to be appended with pre existing data
        self.append = append
        # Flag that specifies to skip generation of data if data pre-exists for the gloss
        self.skip = skip

    def _sanitize(self, glosses: List[str]) -> List[str]:
        return [gloss.rstrip().upper().replace(" ", "_") for gloss in glosses]

    def add_gloss(self, glosses: List[str]):
        self.glosses = list(set(self.glosses+self._sanitize(glosses)))

    def __iter__(self):
        for gloss in self.glosses:
            try:
                yield self.gen_gloss_data(gloss, append=self.append, skip=self.skip)
            except Exception as err:
                print("generator error : ", err)
                raise StopIteration()

    def __repr__(self):
        return f"""Glosses     : {self.glosses}\n
                   frame count : {self.frame_count}\n
                   video count : {self.vid_count}"""

    def __len__(self):
        return len(self.glosses)*self.vid_count

    def gen_seq(self, gloss: str, vid_num: Union[None, int] = None) -> List[np.ndarray]:
        # if not self.frame_count:
        #     raise AttributeError("Frame count not set ")
        cap = cv2.VideoCapture(0)
        result: List[np.ndarray] = []
        i = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_holistic = mp.solutions.holistic
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                try:
                    res = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    draw_landmarks(image, res)
                    result.append(proc_landmarks(res))
                    desc = f"frame : {i+1} of [{gloss}]" if not vid_num else f"frame : {i+1} of [{gloss} : {vid_num}]"
                    image = cv2.putText(cv2.flip(image, 1), desc, (10, 35),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                    i += 1
                except Exception as err:
                    print("Error : ", err)
                cv2.imshow(f"asl-recog Data Collection", image)
            if cv2.waitKey(1) == ord("q") or i >= self.frame_count:
                break
        cap.release()
        cv2.destroyAllWindows()
        return result

    def gen_gloss_data(self, gloss: str, append=False, skip=False) -> str:
        # if not self.vid_count:
        #     raise AttributeError("Video Count not defined")

        # Skips generating the data if data already exists for the gloss
        if skip and gdata_count(self.gloss_dir, gloss) > 0:
            return set_gloss_path(gloss, self.gloss_dir)

        # Appends the data count for each video if there exists previous data
        data_cnt = gdata_count(self.gloss_dir, gloss) if append else 1
        path = set_gloss_path(gloss, self.gloss_dir)
        for i in range(self.vid_count):
            res = self.gen_seq(gloss, i+1)
            df = pd.DataFrame(data=res)
            res_file = "{}/{}.csv".format(path, data_cnt+i)
            df.to_csv(res_file, index=False, header=False)
        return path

    def get_labels(self) -> List[str]:
        return get_all_gloss(self.gloss_dir)

    def get_gdata(self, gloss: str) -> GData:
        return _get_gdata(self.gloss_dir, gloss)

    def get_all_gdata(self) -> GData:
        return _get_all_gdata(self.frame_count, self.gloss_dir)

    def generate(self) -> List[str]:
        return [res for res in self]

    def save_dataset(self):
        gdata, label = self.get_all_gdata()
        df = pd.DataFrame({
            "GlossData": gdata.reshape(
                (len(label), self.frame_count*1596)).tolist(),
            "Label": label
        })
        df.to_csv('dataset.csv', index=False)

    def _transform_gdata(self, x: str) -> np.ndarray:
        return np.fromstring(re.sub(r'[\[\]\s+]', "", x), dtype=float, sep=',').reshape((self.frame_count, 1596))

    def load_dataset(self) -> GData:
        df = pd.read_csv('dataset.csv')
        label: List[str] = df['Label'].tolist()
        gd: np.ndarray = df['GlossData'].to_numpy()
        gdata: np.ndarray = np.asarray([self._transform_gdata(x) for x in gd])
        return gdata, label

    def save_checkpoint(self):
        ckpt = json.dumps(self.__dict__)
        with open('gproc_checkpnt.json', 'w') as file:
            file.write(ckpt)
