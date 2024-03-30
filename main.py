import argparse
import os

from modules.table import parse_image


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    parse_image(args.mode, args.pdf_path, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["connected_component", "textractor", "unstructured", "google_doc"],
    )
    parser.add_argument("--pdf_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    main(args)
