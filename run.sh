#!/bin/bash


conda init bash
conda activate fedml
echo "Starting server"
python server.py &
sleep 4  # Sleep for 3s to give the server enough time to start

for i in `seq 0 4`; do
    echo "Starting client $i"
    python client.py &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
$SHELL