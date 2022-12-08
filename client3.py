# this code is for local client
ll, ul = 80000, 120000
import socket
from time import sleep


MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
MULTICAST_TTL = 2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)

first_iter = True
model_from_server = None
count = 0
client_no = 3
round_no = 0
while True:
    if first_iter:
        sock.sendto(bytes("hello", "utf-8"), (MCAST_GRP, MCAST_PORT))
        first_iter = False
    else:
        import pickle
        model_from_server = pickle.loads(model_from_server)
        ll, ul = 80000+count, 90000+count
        count += 2500

        if round_no == 0:
            features, metrics = model_from_server.myMLP(model_from_server, [round_no, client_no, ll, ul])
        else:
            features, metrics = model_from_server.myMLP([round_no, client_no, ll, ul])
        # send the features to the server
        data = pickle.dumps([features, metrics])
        sock.sendto(bytes(data), (MCAST_GRP, MCAST_PORT))
        print("Features sent, waiting for the model")
        round_no += 1

    # wait for the global server to send a message
    model_from_server = sock.recv(50000000)
    print("New model received...")
    sleep(2)
