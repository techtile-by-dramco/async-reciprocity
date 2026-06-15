import time
from utils.client_com import Client as ComClient
from utils.USRPClient import USRPClient

com = ComClient(args.config_file)
rf_client = USRPClient(args.config_file)

def handle_sync(args):
    rf_client.sync(args.mode)

com.on("SYNC", handle_sync)
com.on("START", handle_start)
com.on("CAL", handle_cal)
com.on("PILOT", handle_pilot)
com.on("STOP", handle_stop)

com.start() # start communication


while com.running:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        pass

com.stop()
com.join()
