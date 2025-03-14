import numpy as np
from logging import error

##################################################
# This class provides tools for conducting
# synchronization protocol with noisy channels.

class Kraus_synch:
    # constructor requires passing only three parameters:
    # dim -> n : number of nodes 
    # phi_a,phi_b : two angles corresponding to the generalized rotations 

    def __init__(self, dim : int , phi_a : float, phi_b : float) -> None:
        self.dim = dim
        self.ch_a,self.ch_b = self.get_all_rot(phi_a,phi_b)

    # This method provides R^{[k]}_{ij}(\phi) operator from the paper 

    def getRotation(self, phi : float, i : int, j : int, k : int) -> np.ndarray:
        ans = np.eye(self.dim)
        ans[i,i] = np.cos(phi)
        ans[i,j] = -np.sin(phi)
        ans[j,j] = np.cos(phi)
        ans[j,i] = np.sin(phi)
        ans[k,k] = 0
        return ans
    
    # This method provides a set of operators representing 
    # noisy channels.

    def get_all_rot(self,phi_a : float, phi_b : float) -> tuple:
        rot_a = [self.getRotation(phi_a,1,2,0)]+[self.getRotation(phi_a,i,i + 1,0) for i in range(2,self.dim-1)]
        rot_b = [self.getRotation(phi_b,0,2,1)]+[self.getRotation(phi_a,i,i + 1,1) for i in range(2,self.dim-1)]

        A1 = np.eye(self.dim)
        B1 = np.eye(self.dim)
        for i in range(len(rot_a)):
            A1 = A1 @ rot_a[i]
            B1 = B1 @ rot_b[i]

            A2 = np.zeros((self.dim,self.dim))
            B2 = np.zeros((self.dim,self.dim))
            A2[1,0] = 1
            B2[0,1] = 1
        return [A1,A2], [B1,B2]

    def make_step(self,rho : np.ndarray, letter :str = "A") -> np.ndarray:
        if letter == "A":
            return self.ch_a[0] @ rho @ self.ch_a[0].conj().T + self.ch_a[1] @ rho @ self.ch_a[1].conj().T
        elif letter == "B":
            return self.ch_b[0] @ rho @ self.ch_b[0].conj().T + self.ch_b[1] @ rho @ self.ch_b[1].conj().T
        else:
            raise error("invalid letter")
        
    # Those two methods are used in order to produce an initial state.
    # Only two of possible initial states are provided.
    # One can easily modified this code to obtain more generalized case.

    def prepare_mixed_state(self) -> np.ndarray:
        return np.eye(self.dim) / self.dim

    def prepare_equally_weighted_state(self) -> np.ndarray:
        st = np.ones(self.dim) / np.sqrt(self.dim)
        return np.kron(st.reshape(-1,1),st)
    
    # This method allows to run the simulation for certain word
    # it should be read from left to write. 
    # Each letter corresponds to the particular channel.

    def run(self,word : str, **kwargs) -> np.ndarray:
        mode = kwargs.get("init","mixed")
        rho = self.prepare_mixed_state() if mode == "mixed" else self.prepare_equally_weighted_state()
        for letter in word:
            rho = self.make_step(rho,letter = letter)
        return rho
    
##################################################