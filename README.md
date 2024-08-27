# ToyNamer
A simple but fun Turkish name generator model

## Streamlit App

To check out the streamlit app, go to this link: https://toynamer.streamlit.app

## Training

To train the model with default hyperparameters and dataset use the following command:
```bash
python src/train.py
```

## Evaluating

If you want to evaluate the trained model weights on evaluation or another dataset use example below:
```bash
python src/evaluate.py \
--val_data_path=data/val_dataset.txt \
--model_path=outputs/train_run_2024-08-22_21-09-19/best.pth
```

## Runnig the app localy

To run the app in your local use the following command:
```bash
streamlit run app/toynamer_app.py
```