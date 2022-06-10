export $(xargs <env_vars.env)

if [ -z ${SERVER_PORT+x} ]; then
  SERVER_PORT=5002
else
  SERVER_PORT="${SERVER_PORT}"
fi

python -m uvicorn retail_multi_model.api.server:app --host 0.0.0.0 --port $SERVER_PORT