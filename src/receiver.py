# this code is for global server
import socket
import struct
import pickle

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
IS_ALL_GROUPS = True

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
if IS_ALL_GROUPS:
    # on this port, receives ALL multicast groups
    sock.bind(('', MCAST_PORT))
else:
    # on this port, listen ONLY to MCAST_GRP
    sock.bind((MCAST_GRP, MCAST_PORT))
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)

sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

while True:
    clients = []
    estimators = []
    models_received = 0
    clf = None
    while True:
        classifier_as_str, address = sock.recvfrom(100000000)
        print("Model received from", address, end=" ")
        clients.append(address)

        classifier = pickle.loads(classifier_as_str)
        models_received += 1
        estimators.append(("client"+str(models_received), classifier))
        print(models_received, "model(s) received")
        if models_received == 4:
            print("Models received")
            break

    from sklearn.ensemble import StackingClassifier
    from sklearn.neural_network import MLPClassifier
    clf = StackingClassifier(estimators=estimators, final_estimator=MLPClassifier(hidden_layer_sizes=(3, 2), random_state=5, learning_rate_init=0.5))

    from model import global_test
    global_test(clf)
    estimators.clear()

    status = input("Do you want to continue? (y/n): ")
    if status == "y":
        model_as_str = pickle.dumps(classifier)
        for client in clients:
            sock.sendto(bytes(model_as_str), client)
        print("Global model sent to clients")
    else:
        exit(0)
