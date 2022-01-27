python tf/server.py &

sleep 1 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect

python tf/client.py &
python tf/client.py &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3` or ultimately `pkill python`
sleep 86400