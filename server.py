# this code is for global server
import socket
import time
import struct
import pickle
import os
import statistics
import basehash
import model
from cryptography.fernet import Fernet
import matplotlib.pyplot as plt
import numpy as np


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

print("Server started, waiting for clients...")
total_no_of_clients = 4
rounds = 15
first_iter = True
round_no = 0
clients = []
accuracies = []
credentials = {x: x for x in range(1, total_no_of_clients + 1)}
for _ in range(rounds):
    weights_arr = []
    biases_arr = []
    features_received = 0
    while first_iter:
        data, addr = sock.recvfrom(50000000)
        userid_hashed, password_hashed = pickle.loads(data)

        hash_fn = basehash.base36()
        user_id = hash_fn.unhash(userid_hashed)
        password = hash_fn.unhash(password_hashed)

        if credentials[user_id] == password:
            clients.append(addr)
            print("\r",len(clients), "client(s) authenticated...",end="")
            time.sleep(0.5)
        if len(clients) == total_no_of_clients:
            break
    while first_iter:
        model_as_str = pickle.dumps(model.Model)
        key = Fernet.generate_key()
        f = Fernet(key)
        model_as_str = f.encrypt(model_as_str)
        encryption_arr = pickle.dumps([key, model_as_str])

        for client in clients:
            sock.sendto(bytes(encryption_arr), client)
        print("\nGlobal model sent!")
        first_iter = False
        break

    print("Round-", round_no+1, "     Receiving features...", sep="", end=" ")
    global_acc, global_loss, global_f1, global_recall = [], [], [], []
    while True:
        # receive features from clients
        data, addr = sock.recvfrom(50000000)
        encryption_arr_from_client = pickle.loads(data)
        data = encryption_arr_from_client[1]
        data = Fernet(encryption_arr_from_client[0]).decrypt(data)
        data = pickle.loads(data)

        weights_arr.append(data[0])
        biases_arr.append(data[1])
        global_acc.append(data[2][0])
        global_loss.append(data[2][1])
        global_f1.append(data[2][2])
        global_recall.append(data[2][3])
        features_received += 1
        if features_received == total_no_of_clients:
            break
    print("done")
    print("Accuracy:", "{:.5f}".format(statistics.mean(global_acc)), end=" ")
    print("Loss:", "{:.5f}".format(statistics.mean(global_loss)), end=" ")
    print("F1Score:", "{:.5f}".format(statistics.mean(global_f1)), end=" ")
    print("Recall:", "{:.5f}".format(statistics.mean(global_recall)), end="\n")

    accuracies.append(statistics.mean(global_acc))

    status = "y"  # input("Do you want to continue? (y/n): ")
    if status == "y":
        # average the features
        zipped_weights = zip(*weights_arr)
        averaged_weights = [sum(feature) / len(feature) for feature in zipped_weights]

        zipped_biases = zip(*biases_arr)
        averaged_biases = [sum(feature) / len(feature) for feature in zipped_biases]

        new_model = model.Model(averaged_weights, averaged_biases)
        model_as_str = pickle.dumps(new_model)

        key = Fernet.generate_key()
        f = Fernet(key)
        encrypted_model = f.encrypt(model_as_str)
        encryption_arr = pickle.dumps([key, encrypted_model])

        for client in clients:
            sock.sendto(bytes(encryption_arr), client)
        print("Global model sent to clients")
        round_no += 1
    else:
        for client in clients:
            sock.sendto(bytes("n", "utf-8"), client)
        exit(0)
else:
    for client in clients:
        sock.sendto(bytes("n", "utf-8"), client)

    x_points = np.array([x+1 for x in range(rounds)])
    y_points = np.array(accuracies)
    from scipy.interpolate import make_interp_spline, BSpline
    xn = np.linspace(x_points.min(), x_points.max(), 300)
    spl = make_interp_spline(x_points, y_points, k=3)  # type: BSpline
    yn = spl(xn)
    plt.plot(xn, yn)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Rounds")
    plt.yticks([x for x in range(40, 100, 10)])

    previous_dir = os.getcwd()
    os.chdir(r"C:\Users\ADMIN\pythonFLProject\outputs")
    plt.savefig("avg_global_acc")
    plt.show()
    plt.close()

    # go back to previous directory
    os.chdir(previous_dir)

    exit(0)
