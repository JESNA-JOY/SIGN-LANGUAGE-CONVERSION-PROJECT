import cv2
import argparse
import os
import string
import time
from gloss_proc import GlossProcess


def main():
    parser = argparse.ArgumentParser(
        description="Cli to Generate asl gloss data")
    parser.add_argument(
        "-g",
        "--gloss",
        help="Gloss name to add data",
    )
    parser.add_argument(
        "-r", "--require", help="Generate gloss from gloss.txt", action="store_true"
    )
    parser.add_argument(
        "-a", "--append", help="Append GlossProc Data if it exists", action="store_true"
    )
    parser.add_argument(
        "-s", "--skip", help="Skip GlossProc Data if already exists", action="store_true"
    )
    parser.add_argument(
        "-c", "--count", help="Add number of gloss [count from 0 or index specified by -i flag ] ie  get glosses[i:n](where i=0 by default) from gloss.txt", type=int
    )
    parser.add_argument(
        "-i", "--index", help="select i th gloss from gloss.txt", type=int
    )
    parser.add_argument(
        "-v", "--vid", help="Add number of videos for each class [default:20]", type=int
    )
    parser.add_argument(
        "-f", "--frame", help="Add number of frames in sequence of each class [default:30]", type=int
    )
    vid_count = parser.parse_args().vid if parser.parse_args().vid else 20
    frame_count = parser.parse_args().frame if parser.parse_args().frame else 24
    if not parser.parse_args().require:
        if parser.parse_args().gloss:
            gloss = parser.parse_args().gloss
        else:
            print("Necessary arguments [ -g or -r ] not supplied\n")
            parser.print_help()
            exit()
        try:
            append_flag = True if parser.parse_args().append else False
            skip_flag = True if parser.parse_args().skip else False
            gp = GlossProcess([gloss], vid_count=vid_count,
                              frame_count=frame_count, append=append_flag, skip=skip_flag)
            res = gp.generate()
            gp.save_checkpoint()
        except Exception as err:
            print(err)
    else:
        try:
            with open('gloss.txt', 'r') as file:
                glosses = file.readlines()
            gloss_len = abs(parser.parse_args().count) if parser.parse_args(
            ).count else len(glosses)
            init_index = abs(
                parser.parse_args().index) if parser.parse_args().index else 0
            if init_index > gloss_len:
                raise IndexError("Index Greater than total number of gloss")
            append_flag = True if parser.parse_args().append else False
            skip_flag = True if parser.parse_args().skip else False
            gp = GlossProcess(glosses[init_index:gloss_len], vid_count=vid_count,
                              frame_count=frame_count, append=append_flag, skip=skip_flag)
            gp.generate()
            gp.save_checkpoint()
        except Exception as err:
            print(err)


if __name__ == "__main__":
    main()
