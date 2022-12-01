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
    features_arr = []
    features_received = 0
    total_no_of_clients = 4
    while True:
        data, addr = sock.recvfrom(50000000)
        clients.append(addr)
        if len(clients) == total_no_of_clients:
            break
    while True:
        import model
        model_as_str = pickle.dumps(model.Model)

        for client in clients:
            sock.sendto(bytes(model_as_str), client)
        print("Global model sent")
        break

    while True:
        # receive features from clients
        data, addr = sock.recvfrom(50000000)
        features_arr.append(pickle.loads(data))
        features_received += 1
        if features_received == total_no_of_clients:
            break

    status = input("Do you want to continue? (y/n): ")
    if status == "y":
        # average the features
        zipped_features = zip(*features_arr)
        averaged_features = [sum(feature) / len(feature) for feature in zipped_features]

        new_model = model.Model(averaged_features)
        model_as_str = pickle.dumps(new_model)

        for client in clients:
            sock.sendto(bytes(model_as_str), client)
        print("Global model sent to clients")
    else:
        exit(0)
