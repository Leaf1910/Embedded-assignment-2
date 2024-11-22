# Import the InferencePipeline object
from inference import InferencePipeline
import cv2


def my_sink(result, video_frame):
    if result.get("output_image"):  # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print(result)  # do something with the predictions of each frame


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="07MTHAw4YHEU2802oGx7",
    workspace_name="drone-dwnqy",
    workflow_id="custom-workflow-4",
    video_reference="Road.mp4",  # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start()  # start the pipeline
pipeline.join()  # wait for the pipeline thread to finish
