# SpotNN2
SpotNN2 ML Models
Vacancy recognition ML solution that utilizes Deep Learning scheme for Computer Vision. 
Models: VGG-16 and 3X64 Node Convolutional Neural Network

Parking Spot Recognition Solution(s):

The parking spot recognition solution(s) are required to identify vacancies or occupancy in
the parking spaces using Geospatial Imagery Analytics methods. It utilizes Computer Vision,
and training Machine Learning models, to extract open parking spots in a parking space
based on data extraction from video feeds from the security cameras.
The recognition solution(s) will be used for object detection to identify all of the parked
vehicles and to verify overlapping of the vehicles with the respective parking spot.
Convolutional Neural Networks (CNNs), that fall under the domain of Deep Learning
Scheme for Computer Vision, were used to analyze each parking spot and to predict
occupancy or vacancy of those spots. The vehicles and objects of interest are located and
classified within certain boundaries. Deep neural networks differentiate the vehicles from
objects that are not vehicles by analyzing a multitude of features. Appropriate artificial
intelligence systems or machine learning models were developed as best fit for the
solution, and were trained using the video feed data. Image segmentation and other
techniques and tasks were undertaken for refinement (as required), to achieve a
satisfactory accuracy of spot vacancy/occupancy detection with the model. The phases
include detection and IoU( Intersection Over Union). Detection decisions are returned as
Boolean. Research and some experimentation was required with the existing popular AI
models to compare results and to identify the best-fitting model for the solution; Fine
tuning and modifications were undertaken.
For the Parking Spot Recognition solution(s), deep learning architecture in CV: VGG-16 was
used for one of two solutions, while the second solution was based on a custom 3X64 Node
Convolutional Neural Network model. The VGG16 model trains on RGB color image dataset
and the 3X64CNN model on grayscale data upon preprocessing; One model can yield
better accuracy/loss than the other, depending on the type/quality/amount of data we are
able to train with (considering the limited resources during the crisis timeframe). Both
models implements the following:
Once a dataset is populated for training the model, it first undergoes Pre-processing where
the images are normalized, converted, etc. to prepare it to be fed into training the model.
Data augmentation could be done to improve the datasets if there is a lack of data in this
timeframe and deal with overfitting to some extent. The validation sets and test sets were
prepared as well and directories set-up accordingly.
The models were then built. The VGG-16 model was fine-tuned using eg. Keras (a
NN-library and API for building and training deep learning models) and the CNN model was
tweaked to fit the project solution. Next, the models were trained using the pre-processed
datasets. The epoch value and other parameters can be tweaked for optimizations.

Predictions were run and statistical tools such as confusion matrix were plotted to assess
the outcome of factors such accuracy and losses. The models were tweaked and fine-tuned
as necessary, and were re-trained. The 3X64-CNN model, during a session, yielded >97%
accuracy on the test set which could be. The prediction outputs along with the spot locators
or indexed locations were returned as JSON and saved. Endpoints were established so that
these output objects can be transmitted and utilized by the front-end.
The model was saved as a .h5 file and was deployed on GeoEvents-ArcGIS Server so that
calls can be received by the front-end of web-application at the prediction endpoint.
The SpotNN_vgg16 model code includes detailed comments, including some
documentations and some important scripts/commands that might be very useful for
future developers to work on or improve on this proof-of-concept solution, and makes the
code very easy to follow. In addition, a demo front-end and Flask web services were used to
pseudo-deploy/host and test the CV solution components prior to pushing it to
github/project.
Some of the technologies, libraries, and/or platforms include OpenCV, Tensorflow,
Matplotlib, numPy, keras, skLearn, Python, and Project Jupyter. Google CoLab was primarily
used to avoid dependency and many other issues that arose initially. Tools and
technologies found via research were used for the project as needed. Polygon selection
tools, spot extractors, and other tools developed by the team supplements the
technological needs for this project.
The scope and features of the recognition solution was dependent on the time constraint,
access to, and availability of resources during the crisis. Parking lots are often empty and
have a limited variety of parked vehicles and hence the dataset to train the model was
limited. The model could benefit from image data extracted with different weather
patterns, shadows, time of days, variety and size of vehicles, and so on, however, the model
can simply be re-trained with a better dataset in the future in order to get better
predictions on a larger variety of instances. The solution(s) are aimed to demonstrate a
proof-of-concept as a deliverable; Additional features or capabilities can be added later or
improved, with sponsors and mentors approval and given additional time and resources.
