export $(xargs <env_vars.env)

if [ -z ${INTERFACE_PORT+x} ]; then
  INTERFACE_PORT=5002
else
  INTERFACE_PORT="${INTERFACE_PORT}"
fi


streamlit run app.py --server.port=$INTERFACE_PORT --server.fileWatcherType none