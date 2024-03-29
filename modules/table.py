import base64
import json
import os
import subprocess

import cv2
import numpy as np
from google.cloud import documentai_v1beta3 as documentai
from google.oauth2 import service_account
from pdf2image import convert_from_path
from textractor import Textractor
from textractor.data.constants import TextractFeatures
from textractor.visualizers.entitylist import EntityList
from unstructured.partition.pdf import partition_pdf

from modules.bounding_box import BoundingBox
from modules.clustering import clustering
from modules.connected_component import (
    find_connected_components,
    get_text_bboxes,
    transform_bboxes_to_points,
)
from modules.visualize import visualize_bounding_boxes


def extracted_by_nougat(file_path: str, output_dir: str) -> int:
    cmd = ["nougat", file_path, "-o", output_dir, "-m", "0.1.0-base", "--no-skipping"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise Exception("Nougat failed")
    return res.returncode


def extracted_by_textractor(pdf_path: str, output_dir: str):
    image = convert_from_path(pdf_path)[0]
    extractor = Textractor(region_name="us-east-1")
    document = extractor.analyze_document(
        file_source=image, features=[TextractFeatures.TABLES], save_image=True
    )
    table = EntityList(document.tables[0])
    output_path = f"{output_dir}/{pdf_path.split('/')[-1].split('.')[0]}.xlsx"
    table[0].to_excel(filepath=output_path)


def extracted_by_unstructured(pdf_path: str, output_dir: str):
    elements = partition_pdf(
        filename=pdf_path, infer_table_structure=True, strategy="hi_res"
    )
    tables = [el for el in elements if el.category == "Table"]
    output_path = f"{output_dir}/{pdf_path.split('/')[-1].split('.')[0]}.html"
    with open(output_path, "w") as f:
        f.write(tables[0].metadata.text_as_html)


def parse_google_doc_response(result_dict: dict) -> dict:
    def parse(data: dict, parsed_result: dict):
        if "layouts" not in parsed_result:
            parsed_result["layouts"] = []

        if isinstance(data, dict):
            if "layout" in data:
                parsed_result["layouts"].append(data["layout"])
            if "image" in data:
                parsed_result["image"] = base64.b64decode(data["image"]["content"])
            for key in data:
                parse(data[key], parsed_result)
        elif isinstance(data, list):
            for item in data:
                parse(item, parsed_result)

    parsed_result = {}
    parse(result_dict, parsed_result)
    return parsed_result


def extracted_by_google_doc(pdf_path: str, output_dir: str):
    credentials = (
        "/Users/junyu.ye/Documents/cloud/keys/norse-breaker-418105-9cc3afa8851f.json"
    )
    project_id = "norse-breaker-418105"
    processor_id = "1fc9eb0761c366a7"

    pdf_name = pdf_path.split("/")[-1].split(".")[0]
    output_json_path = f"{output_dir}/{pdf_name}.json"
    if not os.path.exists(output_json_path):
        credentials = service_account.Credentials.from_service_account_file(credentials)
        client = documentai.DocumentProcessorServiceClient(credentials=credentials)
        location = "us"
        resource_name = client.processor_path(
            project=project_id, location=location, processor=processor_id
        )

        with open(pdf_path, "rb") as f:
            pdf = f.read()

        raw_document = documentai.types.RawDocument(
            content=pdf, mime_type="application/pdf"
        )
        request = documentai.types.ProcessRequest(
            name=resource_name, raw_document=raw_document
        )
        response = client.process_document(request=request)
        result_json = type(response).to_json(response)
        with open(output_json_path, "w") as f:
            f.write(result_json)
        result_dict = json.loads(result_json)
    else:
        result_dict = json.load(open(output_json_path))
    parsed_result = parse_google_doc_response(result_dict)

    array_image = cv2.imdecode(
        np.frombuffer(parsed_result["image"], np.uint8), cv2.IMREAD_COLOR
    )
    bboxes = []
    for layout in parsed_result["layouts"]:
        vertices = layout["boundingPoly"]["vertices"]
        bbox = [(v["x"], v["y"]) for v in vertices]
        bboxes.append(bbox)
    visualize_bounding_boxes(
        array_image, bboxes, f"{output_dir}/{pdf_name}_bbox_google_doc.png"
    )


def extracted_by_connected_component(pdf_path: str, output_dir: str):
    pil_image = convert_from_path(pdf_path)[0]
    raw_image_path = f"{output_dir}/{pdf_path.split('/')[-1].split('.')[0]}.png"
    pil_image.save(raw_image_path)

    color_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    ccs = find_connected_components(binary_image)
    print("Number of all bboxes", len(ccs))
    bboxes = [BoundingBox(cc) for cc in ccs]
    text_bboxes = get_text_bboxes(bboxes)
    print("Number of text bboxes:", len(text_bboxes))
    clusters = clustering(text_bboxes, binary_image.shape[0])
    print("Number of clusters:", len(clusters))
    output_path = f"{output_dir}/{pdf_path.split('/')[-1].split('.')[0]}_text_line.png"
    pts = transform_bboxes_to_points(clusters)
    visualize_bounding_boxes(color_image, pts, output_path)


def parse_image(mode: str, image_path: str, output_dir: str):
    NAME_TO_EXTRACTOR = {
        "textractor": extracted_by_textractor,
        "unstructured": extracted_by_unstructured,
        "google_doc": extracted_by_google_doc,
        "connected_component": extracted_by_connected_component,
    }
    NAME_TO_EXTRACTOR[mode](image_path, output_dir)
