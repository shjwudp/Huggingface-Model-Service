# Huggingface-Model-Service

## GET ON

Install requirements.
```
git clone https://github.com/shjwudp/Huggingface-Model-Service && cd Huggingface-Model-Service
pip install -r requirements.txt
```

Load huggingface model, start HTTP restful service.
```
python generate-server.py --huggingface_model bigscience/bloom-560m --port YOUR_GENERATE_SERVER_PORT
```

Run streamlit server.
```
streamlit run streamlit-server.py -- --backend http://localhost:YOUR_GENERATE_SERVER_PORT/generate
```

View your Streamlit app in your browser.
![image](https://user-images.githubusercontent.com/11439912/195617248-6106e280-daf7-4fe8-bf6b-25ab11003f10.png)
