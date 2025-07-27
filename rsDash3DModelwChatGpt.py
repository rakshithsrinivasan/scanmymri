import os
import tempfile
import dash
import pydicom
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from dash import html
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import itk
from openai import OpenAI
from dotenv import load_dotenv

import dash_vtk
from dash_vtk.utils import to_volume_state

selectedmodel = "ChatGpt"

load_dotenv()

client = OpenAI(api_key = os.getenv("OPEN_API_KEY"))

folder = os.path.join("./", "ct")

def dcm_to_volume(dir_path):
    itk_image = itk.imread(dir_path)
    vtk_image = itk.vtk_image_from_image(itk_image)
    volume_state = to_volume_state(vtk_image)
    return volume_state


demo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
volume_state = dcm_to_volume(folder)


app = dash.Dash(__name__)
server = app.server

vtk_view = dash_vtk.View(
    dash_vtk.VolumeRepresentation(
        [dash_vtk.VolumeController(), dash_vtk.Volume(state=volume_state)]
    )
)

app.layout = html.Div(
    style={
        "height": "100vh",
        "width": "100%",
        "display": "flex",
        "flexDirection": "column"
    },
    children=[
        html.Div(
            style={
                "flex": "1 1 60%",
                "width": "100%",
                "borderBottom": "2px solid #ddd",
                "overflow": "hidden"
            },
            children=vtk_view
        ),

        html.Div(
            style={
                "flex": "1 1 40%",
                "overflowY": "auto",
                "padding": "20px",
                "textAlign": "center",
                "backgroundColor": "#f8f9fa"
            },
            children=[
                html.H2("Analyze MRI Scan", style={"marginBottom": "10px"}),
                html.Button(
                    'Submit Analysis',
                    id='submit-val',
                    n_clicks=0,
                    style={
                        "padding": "10px 20px",
                        "fontSize": "16px",
                        "backgroundColor": "#007bff",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer"
                    }
                ),
                html.Br(),
                html.Br(),

                dcc.Loading(
                    id="loading-spinner",
                    type="circle",
                    color="#007bff",
                    children=html.Div(
                        id='container-button-basic',
                        style={
                            "whiteSpace": "pre-wrap",
                            "textAlign": "left",
                            "margin": "0 auto",
                            "maxWidth": "800px",
                            "maxHeight": "300px",
                            "overflowY": "auto",
                            "padding": "15px",
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #ccc",
                            "borderRadius": "5px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                        },
                        children="Press the button to analyze the MRI scan."
                    )
                )
            ]
        )
    ]
)


def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# def analyzeimage():
#     file_id = create_file("temp_image.png")

#     response = client.responses.create(
#         model="gpt-4o",
#         input=[{
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "what is this image showing an MRI scan of? detail response"},
#                 {
#                     "type": "input_image",
#                     "file_id": file_id,
#                 },
                
#                 {"type": "input_"

#                 },
#             ],
#         }],
#     )

@callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    prevent_initial_call=True
)
def update_output(n_clicks):

    image_list = []
    image_prompt = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_filepath in os.listdir(folder):
            if not uploaded_filepath.lower().endswith(".dcm"):
                print(f"Skipping non-DICOM file: {uploaded_filepath}")
                continue
            file_path = os.path.join(temp_dir, uploaded_filepath)
            print(os.path.join(folder, uploaded_filepath))
            uploaded_file = open(os.path.join(folder, uploaded_filepath), "rb")
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())


            try:
                ds = pydicom.dcmread(file_path, force=True)
                pixel_array = ds.pixel_array
                image_path = file_path.replace(".dcm", ".png")
                cv2.imwrite(image_path, pixel_array)
                file_id = create_file(image_path)

                image_prompt.append({
                    "type": "input_image",
                    "file_id": file_id,
                })

                image = Image.open(image_path)
                image_list.append(image)
            except Exception as e:
                return f"Error reading file {uploaded_file}: {e}"

    if selectedmodel == "Gemini":
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                *image_list,
                "These are sequential MRI Scans of a Human Knee. Please analyze them as a complete scan and summarize any significant findings or abnormalities you observe."
            ]
        )

        print(response.text)
        return response.text

    elif selectedmodel == "ChatGpt":

        # file_id = create_file(file_path)
        # image_prompt.append({
        #     "type": "input_image",
        #     "file_id": file_id,
        # })
        # file_id = create_file("temp_image.png")

        response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is this image showing an MRI scan of? detail response"},
                *image_prompt
    
                # {
                #     "type": "input_image",
                #     "file_id": file_id,
                # },
            ],
        }],
        )

        print(response.output_text)
        return response.output_text







    


if __name__ == "__main__":
    app.run(debug=True)