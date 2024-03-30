import os

from modules.table import parse_image

mode = "connected_component"
pdf_path = "data/table_sample.pdf"
output_dir = "output"


def main():
    os.makedirs(output_dir, exist_ok=True)
    parse_image(mode, pdf_path, output_dir)


if __name__ == "__main__":
    main()
