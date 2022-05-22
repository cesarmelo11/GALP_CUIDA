GALP CUIDA 

CUIDA - Car Unified Inspection for Damage and Dirt with AI, is an AI-based station inspection system. CUIDA creates new, attractive, and custom offers, 
promoting sustainable, cost-efficient cleaning and repatiring services. Connecting stations with GALP network partners using a new shared service paradigm.


****************************************FOLDERS***********************************************************

The used datasets are available in https://ipcapt-my.sharepoint.com/:u:/g/personal/jpbsilva_ipca_pt/EWGHOehOIelKuTwv9y-nPJYB8cWRMw_Jcw61ZtTyeSGBmA?e=tmwaS1
The disposition of the files is the following:

└── data
    ├── damage
    │    ├── images
    │    |    	├── train
    │    |    	├── val
    │    │      ├── test
    │    ├── labels
    │           ├── train
    │           ├── val
    │           ├── test
    │    
    ├── dirt
        ├── images
        |	├── train
        |	├── val
        │      ├── test
        ├── labels
               ├── train
               ├── val
               ├── test 
    


-> Images and correspondent labels used for algorithmic training task. 
-> This folder has two sub-folders, related to the Visual Dirt Inspection and Visual Damage Inspection approaches. 
-> Inside each one, the images are labels are splitted into three different subsets for supervised training (train, validation and test).

-> Labels Format: Class X_CENTER Y_CENTER WIDTH HEIGHT

	Damage Classes:
		-> 0: Scratch
		-> 1: Broken Glass
		-> 2: Deformation
		-> 3: Broken

	Dirt Classes:
		-> 0: Wheels Dirt
		-> 1: Wheels Clean
		-> 2: Lateral Dirt
		-> 3: Lateral Clean
		-> 4: Top Dirt
		-> 5: Top Clean

	X_CENTER Y_CENTER WIDTH HEIGHT: Information regarding the bounding box that delimitates the object to be identified.


*** MODELS *** 

└── models
    ├── damage_best.pt
    |
    ├── dirt_best.pt

-> Pytorch models generated after training task that will be used for real-time inference. The selection was based on the training epoch that 
had the best metrics results. 

-> All models were customized and trained using the YOLOv5 repository, presented in https://github.com/ultralytics/yolov5.


*** SCRIPTS:

└── scripts
    ├── models
    │    
    ├── utils 
    │    
    ├── merge.py 

-> Software component of the system. 

-> models/utils folders: Needed functions and libraries for algorithmic inference.

-> merge.py: Python Script for system validation and operation.


*** CUIDA Report.docx ***

-> Document with the detailed information about the whole system (Approach, Implementation, Results)

*** Demo.mp4 ***

-> Video demonstrator of the system working.


***UI***

-> Video demonstrator and user interface of the proposed design for a future GALP app.


