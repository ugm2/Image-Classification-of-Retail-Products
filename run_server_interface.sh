python -m uvicorn retail_multi_model.api.server:app --host 0.0.0.0 --port 5002 &
streamlit run interface/interface.py --server.port=5003 --server.fileWatcherType none && fg