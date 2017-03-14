# README

This respository contains my Insight Data Science project code. The code is in ```Podium_Keyword_Extraction.py```, and performs topic modeling and sentiment analysis using natural language processing to automatically parse online reviews for the Podium company. File ```Podium_Keyword_Extraction.ipynb``` is a Jupyter notebook containing a detailed explanation of what the code does and how it is used.

Directory ```bokeh_application/``` contains the source code for a Bokeh application called ```keyword_extraction.py```, which can be used to visualize the output of ```Podium_Keyword_Extraction.py```. To run the application, a couple of things must be done first:

* Install the Bokeh library by running ```pip install bokeh``` from the terminal command line.
* Download the ```bokeh_application/``` directory.

In ```bokeh_application/``` there is a directory called ```processed_data/``` which contains the output of ```Podium_Keyword_Extraction.py``` necessary to run the application. 

* Open file ```keyword_extraction.py```, and change the file paths at the top to where the ```processed_data/``` files are located.

To run the application, cd to where the file ```keyword_extraction.py``` is located, and in the terminal type:

* ```bokeh serve --show keyword_extraction.py```

The application will run on a Bokeh server, and can be found at ```http://localhost:5006/keyword_extraction```.
