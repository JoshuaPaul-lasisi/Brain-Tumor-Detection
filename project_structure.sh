project/
├── data/
│   ├── Training/                    # Training dataset
│   └── Testing/                     # Testing dataset
├── model/
│   └── brain_tumor_model.keras      # Saved model file
├── reports/
│   ├── images/                      # Folder for storing plots/images
│   └── summary.md                   # Summary markdown file
├── src/
│   ├── data_preprocessing.py        # Script for data loading and preprocessing
│   ├── model_building.py            # Script for building and training the model
│   ├── utils.py                     # Script for plotting utility functions
│   └── app.py                       # Flask app
├── Dockerfile                       # Dockerfile for creating the container
├── requirements.txt                 # Required packages
└── README.md                        # README file for the project