# this code is for local client
import socket

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
MULTICAST_TTL = 2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)

for _ in range(1):
    from model import myMLP
    classifier, acc = myMLP(80000, 120000)
    print(acc)

    import pickle
    model_as_str = pickle.dumps(classifier)
    sock.sendto(bytes(model_as_str), (MCAST_GRP, MCAST_PORT))
    print("Model sent")

