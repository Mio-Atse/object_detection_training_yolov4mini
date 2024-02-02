In this repository, we are going o learn how to train model on yolov4mini for your own objects from video file. 
## Creating environment in Anaconda (Optional) 

I recommend using environment for libraries. Go to https://www.anaconda.com/ and downloand Anaconda Navigator
After setup, run Anaconda Navigator

On Anaconda Navigator, go to Environments tab.

Create a new environment on Python 3.11.* and name it.

After creating environment, run Anaconda Prompt

If you see (base) in the start of the command line, you must open the environment first.
You can use `conda env list` command to see environments on your computer. You must see your newly created environment in this list.

Use following command to open your newly created environment. Make sure to chang last word with your environment name.

`conda activate YOUR_ENVIRONMENT_NAME`

## How To Start
First, clone repository.

Then go to file using `cd object_detection_training_yolov4mini` in your terminal. (Do not shut down the terminal)

Use `python install -r requirements.txt` code in terminal to download necessary libraries.

Make sure that your video is inside the 'videos' folder with name 'video.mp4'

Run `generate_dataset_from_video`

Examine images folder when code running. If you decide you have saved enough files, press 'q' to finish the process (generally 300 to 1000 images depend on the classified objects)

After creating data, go back to your terminal and run `jupyter notebook .` command.

In jupyter notebook, run `label_dataset.ipynb` according to instructions that given in the file.

Trainign can be take 5 hours depends on your dataset and labels. But within 15 to 30 minutes, you can use newly created file on drive/yolov4-tiny `yolov4-tiny-custom_last.weights` with a certain degree of coherenc to object detection.

Copy `yolov4-tiny-custom_last.weights` to your folder which is `object_detection_training_yolov4mini`

run `yolo_opencv_detect.py`


