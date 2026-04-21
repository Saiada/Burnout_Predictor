Model Files (Download Required)

Due to GitHub's 100MB file size limit, trained model files are hosted on Google Drive.

Download Instructions:

1. Download models from: https://drive.google.com/drive/folders/1iNjq7b7_loihbMl3YruxJw6Fk2wSWKGr?usp=sharing
2. Extract the files to the project root folder
3. Verify these files are present:
   - pvt_lstm_model.h5
   - sart_2back_rf_model.pkl
   - saved_models/bert_burnout_model/ (folder)

Setup & Run:

bash:

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py
