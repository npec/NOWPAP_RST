# NOWPAP Training Course on Remote Sensing Data Analysis
*** 

---
The Special Monitoring and Coastal Environmental Assessment Regional Activity Centre (CEARAC), one of the four Regional Activity Centres (RACs) of the Northwest Pacific Action Plan (NOWPAP), offers an intensive training course on the analysis of remote sensing data for marine environment conservation, with a focus on the Northwest Pacific Ocean. 
The course consists of the following 2 themes;  

### Theme 1 ###
- Monitoring and Assessment of Water Quality by Ocean Color Remote Sensing

### Theme 2 ###
- Mapping Seagrass by Optical Sensors  

Both of themes include lectures by experts and hands-on tutorial sessions on analysis of satellite data for the participants to gain useful skills and knowledge to utilize remote sensing data for monitoring and assessing the coastal and marine environment. 
Through the lectures and hands-on sessions, CEARAC also introduces our recently developed monitoring and assessment tools (Seagrass Mapper, Seagrass Trainer, and Global Eutrophication Watch).

---
These courses are targeted at young scientists, postgraduate students, professional researchers and local government officers working in the fields of marine sciences in the Northwest Pacific Region and adjacent area. 


---
## Prerequisites: (for local environment only!)
---
All hands-on use Jupyter Notebook to run Python code.   
Please install Jupyter Notebook from this [link](https://www.anaconda.com/products/individual).

Please also check this [Jupyter notebook](https://github.com/npec/NOWPAP_RST/anaconda-install-setup.ipynb) which explains how to have Jupyter notebook installed on different platforms.

After the installion is done, please install required modules using the [requirements.txt](https://github.com/npec/NOWPAP_RST/requirements.txt)   
To do that (Windows)
- Open Anaconda Prompt.
- Type `pip install -r path_to_requirements/requirements.txt` then `enter`

On MacOS, you should be able to install the requirements through the terminal by launching it using ANACONDA NAVIGATOR. 
For MacOS users, it is worth checking this [help page](https://docs.anaconda.com/anaconda/install/mac-os/).

Most MacOS come with Python2 pre-installed. In this tutorial we are going to use Python3.  
To avoing any conflict between the Python from the system and the one we install, it is recommended to create an isolated virtual environment; a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages.  
Here we are going to create our environment using `anaconda` and we will call it `nowpap_rst` short for NOWPAP Remote Sensing Training.  
- From the ANACONDA NAVIGATOR click `Environments` 
- Under the `base (root)` push the play button and then select `Open Terminal`.  
- Inside terminal enter `conda env create -f path_to_environment_file\environment.yml` then `enter`.
- After the installation is completed, in the ANACONDA NAVIGATOR `Environments` select `nowpap_rst`.
- Under the play button select `Open with Jupyter Notebook`.
This will launch the notebook from the created environment.

If you are unable to have the system working on your computer, please let us know and we can do it together during the hands-on.

---
## Organizer:

[NOWPAP CEARAC](http://cearac.nowpap.org/)

---
## For Inquiries
Genki Terauchi (terauchi[at]npec.or.jp)

---
## License and terms of usage
All the python codes provided here are distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).
