import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import numpy as np

from utils.homomorphic import TrustedAuthority, FHEServer, FHEClient, contains_file

storage_path = "./he_storage"
public_key_path = "public_context.bin"
context_path = "secret_context.bin"
filename_1_v, filename_1_n = "client1_v.bin", "client1_n.bin"
filename_2_v, filename_2_n = "client2_v.bin", "client2_n.bin"
filename_3_v, filename_3_n = "client3_v.bin", "client3_n.bin"
filename_4_v, filename_4_n = "client4_v.bin", "client4_n.bin"
filename_5_v, filename_5_n = "client5_v.bin", "client5_n.bin"
filename_summed_w = "summed_weights.bin"
filename_summed_n = "summed_n.bin"


def gen_vector(size=6707):
    return 4 * np.random.rand(size) - 2


def gen_n(minimum=10, maximum=50):
    return np.random.randint(minimum, maximum + 1)


def main(vector_size=6707):
    plain_vector1 = gen_vector(vector_size)
    plain_vector2 = gen_vector(vector_size)
    plain_vector3 = gen_vector(vector_size)
    plain_vector4 = gen_vector(vector_size)
    plain_vector5 = gen_vector(vector_size)

    num1 = gen_n()
    num2 = gen_n()
    num3 = gen_n()
    num4 = gen_n()
    num5 = gen_n()

    print(num1, num2, num3, num4, num5)

    plain_start = time.time()
    plain_agg = 1 / (num1 + num2 + num3 + num4 + num5) * (
            num1 * plain_vector1 + num2 * plain_vector2 + num3 * plain_vector3 + num4 * plain_vector4 + num5 * plain_vector5)
    plain_end = time.time()

    if contains_file(storage_path, public_key_path) and contains_file(storage_path, context_path):
        print("Found public key and context")
    else:
        print("Public key and context are not found. Generating...")
        ta = TrustedAuthority(poly_modulus_degree=2 ** 13, galois=True, relin=True)
        ta.create_context()
        ta.serialize_context()

    fhe_server = FHEServer()
    client = FHEClient()

    print("[Clients] Encrypting...")
    enc_start = time.time()
    encrypted_vector1 = client.encrypt(num1 * plain_vector1)
    enc_end = time.time()
    enc_num_start = time.time()
    encrypted_num1 = client.encrypt(np.array([num1]))
    enc_num_end = time.time()
    encrypted_vector2 = client.encrypt(num2 * plain_vector2)
    encrypted_num2 = client.encrypt(np.array([num2]))
    encrypted_vector3 = client.encrypt(num3 * plain_vector3)
    encrypted_num3 = client.encrypt(np.array([num3]))
    encrypted_vector4 = client.encrypt(num4 * plain_vector4)
    encrypted_num4 = client.encrypt(np.array([num4]))
    encrypted_vector5 = client.encrypt(num5 * plain_vector5)
    encrypted_num5 = client.encrypt(np.array([num5]))

    print("[Clients] Serializing...")
    ser_start = time.time()
    client.serialize_ciphertext(encrypted_vector1, filename_1_v)
    ser_end = time.time()
    ser_n_start = time.time()
    client.serialize_ciphertext(encrypted_num1, filename_1_n)
    ser_n_end = time.time()
    client.serialize_ciphertext(encrypted_vector2, filename_2_v)
    client.serialize_ciphertext(encrypted_num2, filename_2_n)
    client.serialize_ciphertext(encrypted_vector3, filename_3_v)
    client.serialize_ciphertext(encrypted_num3, filename_3_n)
    client.serialize_ciphertext(encrypted_vector4, filename_4_v)
    client.serialize_ciphertext(encrypted_num4, filename_4_n)
    client.serialize_ciphertext(encrypted_vector5, filename_5_v)
    client.serialize_ciphertext(encrypted_num5, filename_5_n)

    print("[Server] Loading ciphertexts")
    load_start = time.time()
    encrypted_vector1 = fhe_server.load_ciphertext(filename_1_v)
    load_end = time.time()
    load_n_start = time.time()
    encrypted_num1 = fhe_server.load_ciphertext(filename_1_n)
    load_n_end = time.time()

    encrypted_vector2 = fhe_server.load_ciphertext(filename_2_v)
    encrypted_num2 = fhe_server.load_ciphertext(filename_2_n)
    encrypted_vector3 = fhe_server.load_ciphertext(filename_3_v)
    encrypted_num3 = fhe_server.load_ciphertext(filename_3_n)
    encrypted_vector4 = fhe_server.load_ciphertext(filename_4_v)
    encrypted_num4 = fhe_server.load_ciphertext(filename_4_n)
    encrypted_vector5 = fhe_server.load_ciphertext(filename_5_v)
    encrypted_num5 = fhe_server.load_ciphertext(filename_5_n)

    encrypted_weights = [encrypted_vector1, encrypted_vector2, encrypted_vector3, encrypted_vector4, encrypted_vector5]
    encrypted_n = [encrypted_num1, encrypted_num2, encrypted_num3, encrypted_num4, encrypted_num5]

    print("[Server] Performing encrypted FedAvg")
    summed_weights, summed_n, times = fhe_server.fedavg(encrypted_weights, encrypted_n)

    print("[Server] Serializing results")
    ser_sw_start = time.time()
    fhe_server.serialize_ciphertext(summed_weights, filename_summed_w)
    ser_sw_end = time.time()
    ser_sn_start = time.time()
    fhe_server.serialize_ciphertext(summed_n, filename_summed_n)
    ser_sn_end = time.time()

    print("Trying to decrypt on the server side (the server must not be able to decrypt)...")
    try:
        print(summed_weights.decrypt())
    except:
        print("[Server] Cannot decrypt!")

    print("[Client] Loading encrypted results...")
    load_sw_start = time.time()
    summed_weights = client.load_ciphertext(filename_summed_w)
    load_sw_end = time.time()
    load_sn_start = time.time()
    summed_n = client.load_ciphertext(filename_summed_n)
    load_sn_end = time.time()

    print("[Client] Decrypting...")
    decrypt_w_start = time.time()
    weights = np.array(summed_weights.decrypt())
    decrypt_w_end = time.time()
    decrypt_n_start = time.time()
    n = summed_n.decrypt()[0]
    decrypt_n_end = time.time()
    aggregated = weights / n

    print("Are plaintext and encrypted aggregation almost equal?", almost_equal(aggregated, plain_agg, 5))
    print("NUM CLIENTS:", 5)
    print("Vector Size:", len(plain_vector1), "\n\n")
    print("[CLIENT - WEIGHTS ENCRYPTION]", enc_end - enc_start, "seconds")
    print("[CLIENT - NUM SAMPLES ENCRYPTION]", enc_num_end - enc_num_start, "seconds")
    print("[CLIENT - WEIGHTS SERIALIZATION]", ser_end - ser_start, "seconds")
    print("[CLIENT - NUM SAMPLES SERIALIZATION]", ser_n_end - ser_n_start, "seconds\n\n")
    print("[SERVER - WEIGHTS DE-SERIALIZATION]", load_end - load_start, "seconds")
    print("[SERVER - NUM SAMPES DE-SERIALIZATION]", load_n_end - load_n_start, "seconds")
    print("[SERVER - WEIGHTS ADD]", times["add_w"], "seconds")
    print("[SERVER - NUM SAMPLES ADD]", times["add_n"], "seconds")
    print("[SERVER - TOTAL FEDAVG]", times["total"], "seconds")
    print("[SERVER - SUMMED WEIGHT SERIALIZATION]", ser_sw_end - ser_sw_start, "seconds")
    print("[SERVER - SUMMED NUM SAMPLES SERIALIZATION]", ser_sn_end - ser_sn_start, "seconds\n\n")
    print("[CLIENT - SUMMED WEIGHT DE-SERIALIZATION]", load_sw_end - load_sw_start, "seconds")
    print("[CLIENT - SUMMED NUM SAMPLES DE-SERIALIZATION]", load_sn_end - load_sn_start, "seconds")
    print("[CLIENT - SUMMED WEIGHT DECRYPTION]", decrypt_w_end - decrypt_w_start, "seconds")
    print("[CLIENT - SUMMED NUM SAMPLES DECRYPTION]", decrypt_w_end - decrypt_w_start, "seconds")
    print("[CLIENT - SUMMED NUM SAMPLES DECRYPTION]", decrypt_n_end - decrypt_n_start, "seconds\n\n")

    total_server_time = ser_sn_end - ser_sn_start + ser_sw_end - ser_sw_start + times[
        "total"] + load_n_end - load_n_start + load_end - load_start
    total_client_time = decrypt_n_end - decrypt_n_start + decrypt_w_end - decrypt_w_start + decrypt_w_end - decrypt_w_start + load_sn_end - load_sn_start + load_sw_end - load_sw_start + ser_n_end - ser_n_start + ser_end - ser_start + enc_num_end - enc_num_start + enc_end - enc_start
    print("[TIME ON SERVER]", total_server_time, "seconds")
    print("[TIME ON CLIENT]", total_client_time, "seconds")
    print("[TOTAL ENCRYPTED AGGREGATION TIME]", total_server_time + total_client_time, "seconds")
    print("[PLAINTEXT AGGREGATION TIME]", plain_end - plain_start, "seconds")


def almost_equal(vec1, vec2, precision):
    if len(vec1) != len(vec2):
        return False

    total_loss = 0
    upper_bound = pow(10, -precision)
    for v1, v2 in zip(vec1, vec2):
        total_loss += (abs(v1 - v2))
        if abs(v1 - v2) > upper_bound:
            return False
    print("Total FHE Loss:", total_loss, "and Mean Loss:", total_loss / len(vec1))
    return True


if __name__ == "__main__":
    main()
