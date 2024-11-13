BRAIN-TUMOR-DETECTION/
├── data/
│   ├── processed/                   # processed dataset
│   ├── Training/                    # Training dataset
│   └── Testing/                     # Testing dataset
├── model/
│   └── brain_tumor_model.keras      # Saved model file
├── notebooks/
│   └── notebook.ipynb
├── reports/
│   ├── images/                      # Folder for storing plots/images
│   └── summary.md                   # Summary markdown file
├── src/
│   ├── data_preprocessing.py        # Script for data loading and preprocessing
│   ├── model_building.py            # Script for building and training the model
│   ├── utils.py                     # Script for plotting utility functions
│   └── flask_app.py                       # Flask app
│   └── streamlit_app.py                       # Streamlit app
├── static/
│   ├── css/
│       └── styles.css
│   ├── js/
│       └── scripts.js
├── templates/
│   ├── index.html
├── Dockerfile                       # Dockerfile for creating the container
├── LICENSE 
├── project_structure.sh 
├── requirements.txt                 # Required packages
└── README.md                        # README file for the project