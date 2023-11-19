import os
import time
from typing import List, Union, Tuple, Dict

import numpy as np
import tenseal as ts


def contains_file(folder_path: str, file_name: str) -> bool:
    """
    Checks if a file ecists in a folder or its subfolders

    Args:
        folder_path (str): Path to the folder to search.
        file_name: Name of the file to search for.

    Returns:
        bool: True if file exists, False otherwise.
    """
    for root, dirs, files in os.walk(folder_path):
        if file_name in files:
            return True
    for root, dirs, files in os.walk(f".{folder_path}"):
        if file_name in files:
            return True
    return False


class TrustedAuthority:
    """
    Trusted Authority representation that can create, serialize and distribute homomorphic encryption contexts.

    Args:
        scheme (ts.SCHEME_TYPE): The homomorphic encryption scheme to be used. Default is `ts.SCHEME_TYPE.CKKS`.
        poly_modulus_degree (int): The polynomial modulus degree. Determines the size of the ciphertext and plaintext
                                   data. Default is `8192`.
        coeff_mod_bit_sizes (List[int]): List of bit sizes for the coefficients in the polynomial modulus. Determines
                                         the levels of multiplicative depth supported in the encrypted data.
                                         Default is `[60, 40, 40, 60]`.
        global_scale (Union[int, None]): Scaling factor used to prevent precision loss in encrypted computations.
                                         Default is `2 ** 40`.
        galois (bool): Indicates whether to support Galois operations in the context or not. Default is `True`.
        relin (bool): Indicates whether to support relinearization operations in the context or not. Default is `True`.
    """
    def __init__(self,
                 scheme: ts.SCHEME_TYPE = ts.SCHEME_TYPE.CKKS,
                 poly_modulus_degree: int = 8192,
                 coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
                 global_scale: Union[int, None] = 2**40,
                 galois: bool = True,
                 relin: bool = True
                 ):
        self.scheme = scheme
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        self.galois = galois
        self.relin = relin
        self.context: Union[None, ts.Context] = None

    def create_context(self):
        """
        Creates a homomorphic encryption context based on the specified parameters.
        """
        context = ts.context(
            self.scheme, poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        if self.global_scale is not None:
            context.global_scale = self.global_scale
        if self.galois:
            context.generate_galois_keys()
        if self.relin:
            context.generate_relin_keys()
        self.context = context

    def serialize_context(self, storage_path: str = "./he_storage",
                          public_file_path: str = "public_context.bin",
                          secret_file_path: str = "secret_context.bin") -> None:
        """
        Serialize the homomorphic encryption context to a file.

        Args:
            storage_path (str): Path to the folder where context files are saved.
            public_file_path (str): Path for the public part of the context.
            secret_file_path (str): Path for the secret part of the context.
        """
        if self.context is not None:
            serialized_public_context = self.context.serialize(
                save_public_key=True, save_galois_keys=True, save_relin_keys=True, save_secret_key=False
            )
            serialized_secret_context = self.context.serialize(
                save_public_key=True,
                save_galois_keys=True, save_relin_keys=True, save_secret_key=True
            )

            # try to save the serialized contexts to the specified paths.
            # if the paths do not exist, attempt to save in the current directory's subfolder.
            try:
                with open(f"{storage_path}/{public_file_path}", "wb") as f:
                    f.write(serialized_public_context)

                with open(f"{storage_path}/{secret_file_path}", "wb") as f:
                    f.write(serialized_secret_context)
            except FileNotFoundError:
                try:
                    with open(f".{storage_path}/{public_file_path}", "wb") as f:
                        f.write(serialized_public_context)

                    with open(f".{storage_path}/{secret_file_path}", "wb") as f:
                        f.write(serialized_secret_context)
                except FileNotFoundError:
                    raise FileNotFoundError("[TA] Cannot serialize context")
        else:
            print("[TA] Cannot Serialize an empty context")


class FHEServer:
    """
    Server representation that performs homomorphic encryption operations.

    Args:
        storage_path (str): The path where homomorphic encryption data is stored.
            Defaults to "./he_storage".
        context_path (str): The path to the file from which the public encryption context
            is loaded. Defaults to "public_context.bin".
    """
    def __init__(self,
                 storage_path: str = "./he_storage",
                 context_path: str = "public_context.bin",
                 context: Union[None, ts.Context] = None
                 ):
        self.storage_path = storage_path
        if context is not None:
            self.__context = context
        else:
            self.__context = self.load_context(context_path)

    def load_context(self, context_path: str):
        """
        Load a homomorphic encryption context from a file.

        Args:
            context_path (str): Path to the context file.

        Returns:
            ts.Context: Loaded context.
        """
        if not contains_file(self.storage_path, context_path):
            raise FileNotFoundError(
                f"[Server] Cannot find the specified context ({context_path}) in {self.storage_path}!")
        else:
            print("[Server] Context found")
            try:
                with open(f"{self.storage_path}/{context_path}", "rb") as f:
                    context = f.read()
            except FileNotFoundError:
                try:
                    with open(f".{self.storage_path}/{context_path}", "rb") as f:
                        context = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError("[Server] Cannot read context")

        return ts.context_from(context)

    def load_ciphertext(self, file_path: str) -> ts.CKKSVector:
        """
        Load a ciphertext from a file.

        Args:
            file_path (str): Path to the ciphertext file.

        Returns:
            ts.CKKSVector: Loaded ciphertext.
        """
        try:
            with open(f"{self.storage_path}/{file_path}", "rb") as f:
                encrypted = f.read()
        except FileNotFoundError:
            try:
                with open(f".{self.storage_path}/{file_path}", "rb") as f:
                    encrypted = f.read()
            except FileNotFoundError:
                raise FileNotFoundError("[Server] Cannot read ciphertext")
        enc_vector = ts.ckks_vector_from(self.__context, encrypted)
        return enc_vector

    def serialize_ciphertext(self, vector: ts.CKKSVector, file_path: str) -> None:
        """
        Serialize a ciphertext to a file.

        Args:
            vector (ts.CKKSVector): The encrypted vector.
            file_path (str): The path to serialize the encrypted vector.
        """
        serialized_vector = vector.serialize()
        try:
            with open(f"{self.storage_path}/{file_path}", "wb") as f:
                f.write(serialized_vector)
        except FileNotFoundError:
            try:
                with open(f".{self.storage_path}/{file_path}", "wb") as f:
                    f.write(serialized_vector)
            except FileNotFoundError:
                raise FileNotFoundError("[Server] Cannot Serialize ciphertext")

    def encrypt(self, vector: np.ndarray) -> ts.CKKSVector:
        """
        Encrypts a given vector using the current context.

        Args:
            vector (np.ndarray): The input vector to be encrypted.

        Returns:
            ts.CKKSVector: The encrypted vector.
        """
        return ts.ckks_vector(self.__context, vector)

    def add(self, vectors: List[Union[ts.CKKSVector, np.ndarray]]) -> ts.CKKSVector:
        """
        Computes the element-wise addition of the given list of vectors.

        Args:
            vectors (List[Union[ts.CKKSVector, np.ndarray]]): A list of vectors (either encrypted or plain)
                    to be added together.

        Returns:
            ts.CKKSVector: The result of the element-wise addition.
        """
        res = vectors[0]
        for vector in vectors[1:]:
            res = res.add(vector)
        return res

    def subtract(self, vectors: List[Union[ts.CKKSVector, np.ndarray]]) -> ts.CKKSVector:
        """
        Computes the element-wise subtraction of the given list of vectors.

        Args:
            vectors (List[Union[ts.CKKSVector, np.ndarray]]): A list of vectors (either encrypted or plain)
                from which subsequent vectors will be subtracted.

        Returns:
            ts.CKKSVector: The result of the element-wise subtraction.
        """
        res = vectors[0]
        for vector in vectors[1:]:
            res = res.sub(vector)
        return res

    def mul(self, vectors: List[Union[ts.CKKSVector, np.ndarray]]) -> ts.CKKSVector:
        """
        Computes the element-wise multiplication of the given list of vectors.

        Args:
            vectors (List[Union[ts.CKKSVector, np.ndarray]]): A list of vectors (either encrypted or plain)
                to be multiplied together.

        Returns:
            ts.CKKSVector: The result of the element-wise multiplication.
        """
        res = vectors[0]
        for vector in vectors[1:]:
            res = res.mul(vector)
        return res

    def scalar_mul(self, vector: ts.CKKSVector, scalar: int) -> ts.CKKSVector:
        """
        Multiplies a given encrypted vector by a scalar value.

        Args:
            vector (ts.CKKSVector): The encrypted vector to be multiplied.
            scalar (int): The scalar value.

        Returns:
            ts.CKKSVector: The result of multiplying the encrypted vector by the scalar.
        """
        res = vector.mul(scalar)
        return res

    def pow(self, vector: ts.CKKSVector, power: int) -> ts.CKKSVector:
        """
        Raises the given encrypted vector to a specified power.

        Args:
            vector (ts.CKKSVector): The encrypted vector to be exponentiated.
            power (int): The power/exponent.

        Returns:
            ts.CKKSVector: The result of raising the encrypted vector to the specified power.
        """
        res = vector.pow(power)
        return res

    def fedavg(self, weights: List[ts.CKKSVector],
               samples: List[ts.CKKSVector]) -> Tuple[ts.CKKSVector, ts.CKKSVector, Dict[str, float]]:
        """
        Perform federated averaging on encrypted weights and samples.

        Args:
            weights: List of encrypted weights.
            samples: List of encrypted samples.

        Returns:
            Tuple: A tuple containing encrypted summed weights, summed samples, and a dictionary with processing times.
        """
        add_n_start = time.time()
        summed_samples = self.add(samples)
        add_n_end = time.time()

        add_w_start = time.time()
        summed_w = self.add(weights)
        add_w_end = time.time()

        times = {"add_n": add_n_end - add_n_start, "add_w": add_w_end - add_w_start, "total": add_w_end - add_n_start}
        return summed_w, summed_samples, times


class FHEClient:
    def __init__(self,
                 storage_path: str = "./he_storage",
                 context_path: str = "secret_context.bin",
                 context: Union[None, ts.Context] = None
                 ):
        self.storage_path = storage_path
        if context is not None:
            self.__context = context
        else:
            self.__context = self.load_context(context_path)

    def load_context(self, context_path: str):
        if not contains_file(self.storage_path, context_path):
            raise FileNotFoundError(
                f"[Client] Cannot find the specified context ({context_path}) in {self.storage_path}!")
        else:
            print("[Client] Context found")
            try:
                with open(f"{self.storage_path}/{context_path}", "rb") as f:
                    context = f.read()
            except FileNotFoundError:
                try:
                    with open(f".{self.storage_path}/{context_path}", "rb") as f:
                        context = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError("[Client] Cannot load context")
        return ts.context_from(context)

    def encrypt(self, vector: np.ndarray) -> ts.CKKSVector:
        return ts.ckks_vector(self.__context, vector)

    def serialize_ciphertext(self, vector: ts.CKKSVector, file_path: str) -> None:
        serialized_vector = vector.serialize()
        try:
            with open(f"{self.storage_path}/{file_path}", "wb") as f:
                f.write(serialized_vector)
        except FileNotFoundError:
            try:
                with open(f".{self.storage_path}/{file_path}", "wb") as f:
                    f.write(serialized_vector)
            except FileNotFoundError:
                raise FileNotFoundError("[Client] Cannot Serialize ciphertext")

    def load_ciphertext(self, file_path: str) -> ts.CKKSVector:
        try:
            with open(f"{self.storage_path}/{file_path}", "rb") as f:
                encrypted = f.read()
        except FileNotFoundError:
            try:
                with open(f".{self.storage_path}/{file_path}", "rb") as f:
                    encrypted = f.read()
            except FileNotFoundError:
                raise FileNotFoundError("[Client] Cannot read ciphertext")
        enc_vector = ts.ckks_vector_from(self.__context, encrypted)
        return enc_vector
