import sys
import argparse

from anylearning.anylabeling2yolo import AnyLabeling2YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        type=str,
        help="Please input the path of the anylabeling json files.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        nargs="?",
        default=0.1,
        help="Please input the validation dataset size, for example 0.1.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        nargs="?",
        default=0.1,
        help="Please input the test dataset size, for example 0.1.",
    )
    parser.add_argument(
        "--to_seg",
        action="store_true",
        help="Convert to YOLOv5/v8 segmentation dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Please input the output directory for the converted YOLO dataset.",
    )
    args = parser.parse_args(sys.argv[1:])

    convertor = AnyLabeling2YOLO(args.json_dir, to_seg=args.to_seg)
    convertor.convert(output_dir=args.output_dir, val_size=args.val_size, test_size=args.test_size)

