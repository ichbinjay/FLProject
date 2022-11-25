# this code is for local client
ll, ul = 40000, 80000
import socket

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
MULTICAST_TTL = 2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)

first_iter = True
model_from_server = None
while True:
    if first_iter:
        from model import myMLP
        classifier, acc = myMLP(ll, ul)
        first_iter = False
    else:
        import pickle
        classifier = pickle.loads(model_from_server)
        from model import local_test
        classifier, acc = local_test(ll, ul, classifier)

    print(acc)
    import pickle

    model_as_str = pickle.dumps(classifier)
    sock.sendto(bytes(model_as_str), (MCAST_GRP, MCAST_PORT))
    print("Model sent")

    # wait for the global server to send a message
    model_from_server = sock.recv(100000000)
    print("Model received")

