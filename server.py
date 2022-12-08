# this code is for global server
import socket
import struct
import pickle
import statistics

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


total_no_of_clients = 4
rounds = 13
first_iter = True
round_no = 0
clients = []
accuracies = []
for _ in range(rounds):
    features_arr = []
    features_received = 0
    while first_iter:
        data, addr = sock.recvfrom(50000000)
        clients.append(addr)
        if len(clients) == total_no_of_clients:
            break
    while first_iter:
        import model

        model_as_str = pickle.dumps(model.Model)

        for client in clients:
            sock.sendto(bytes(model_as_str), client)
        print("Global model sent")
        first_iter = False
        break

    print("Round-", round_no, "     Receiving features...", sep="", end=" ")
    global_acc, global_loss, global_f1, global_recall = [], [], [], []
    while True:
        # receive features from clients
        data, addr = sock.recvfrom(50000000)
        data = pickle.loads(data)
        features_arr.append(data[0])
        global_acc.append(data[1][0])
        global_loss.append(data[1][1])
        global_f1.append(data[1][2])
        global_recall.append(data[1][3])
        features_received += 1
        if features_received == total_no_of_clients:
            break
    print("done")
    print("Accuracy:", "{:.5f}".format(statistics.mean(global_acc)), end=" ")
    print("Loss:", "{:.5f}".format(statistics.mean(global_loss)), end=" ")
    print("F1Score:", "{:.5f}".format(statistics.mean(global_f1)), end=" ")
    print("Recall:", "{:.5f}".format(statistics.mean(global_recall)), end="\n")

    accuracies.append(statistics.mean(global_acc))

    status = "y" # input("Do you want to continue? (y/n): ")
    if status == "y":
        # average the features
        zipped_features = zip(*features_arr)
        averaged_features = [sum(feature) / len(feature) for feature in zipped_features]

        new_model = model.Model(averaged_features)
        model_as_str = pickle.dumps(new_model)

        for client in clients:
            sock.sendto(bytes(model_as_str), client)
        print("Global model sent to clients")
        round_no += 1
    else:
        for client in clients:
            sock.sendto(bytes("n", "utf-8"), client)
        exit(0)
else:
    for client in clients:
        sock.sendto(bytes("n", "utf-8"), client)
    import matplotlib.pyplot as plt
    import numpy as np

    xpoints = np.array([x for x in range(rounds)])
    ypoints = np.array(accuracies)

    plt.plot(xpoints, ypoints)
    plt.yticks([0,  40, 50, 70, 80, 90])

    import os
    previous_dir = os.getcwd()
    os.chdir(r"C:\Users\ADMIN\pythonFLProject\outputs")
    plt.savefig("avg_global_accs")
    plt.close()
    # go to previous directory
    os.chdir(previous_dir)

    exit(0)
