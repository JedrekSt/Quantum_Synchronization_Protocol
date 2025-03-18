import numpy as np
import matplotlib.pyplot as plt

class SynchronizingWalk:
    def __init__(self,d : int, angle : float) -> None :
        self.dim = d
        #self.U = self.OrdinaryStep() @ self.Coin(angle)
        self.U = self.Coin(angle)
        self.S = self.SynchStep()
        self.Evo = self.S @ self.U 
        #self.Evo = self.S
    
    def make_step_forward(self,rho : np.ndarray,letter : str) -> np.ndarray:
        if letter == "A":
            c_state = np.array([[1,0],[0,0]])
        elif letter == "B":
            c_state = np.array([[0,0],[0,1]])
        else: 
            raise Exception("No letter specified")
        rho = np.kron(rho,c_state)
        rho = self.Evo @ rho @ self.Evo.conj().T
        return rho[::2,::2] + rho[1::2,1::2] 

    def OrdinaryStep(self) -> np.ndarray:
        s1 = np.roll(np.eye(self.dim), 1,0 )
        s1 = np.kron(s1,np.array([[1,0],[0,0]]))
        s2 = np.roll(np.eye(self.dim), -1,0)
        s2 = np.kron(s2,np.array([[0,0],[0,1]]))
        return s1 + s2
    
    def Coin(self,angle : float) -> np.ndarray:
        c_ =  np.array([
            [np.cos(angle),np.sin(angle)],
            [-np.sin(angle),np.cos(angle)]
        ])
        return np.kron(np.eye(self.dim),c_)
    
    def SynchStep(self) -> np.ndarray:
        s1 = np.eye(self.dim,k = -1)
        s1 = np.kron(s1,np.array([[1,0],[0,0]]))

        s2 = np.eye(self.dim,k = 1)
        s2[1,2] = 0
        s2[self.dim - 1 : 0] = 1
        s2 = np.kron(s2,np.array([[0,0],[0,1]]))

        S = s1 + s2
        S[0, 5] = 1
        S[2*self.dim - 1,1] = 1
        S[3,2*(self.dim-1)] = 1

        return S
    
def run_simulation(word : str, dim : int , angle : float) -> np.ndarray:
    rho = np.ones((dim,dim)) / dim
    #rho = np.zeros((dim,dim))
    #rho[dim // 2,dim // 2] = 1
    model = SynchronizingWalk(dim,angle)
    data = [np.diag(rho)]
    n = len(word)
    for i in range(n):
        letter = word[n - 1 - i]
        rho = model.make_step_forward(rho,letter)
        data.append(np.diag(rho))
    return np.array(data)

if __name__ == "__main__*":
    n = 10
    word = ("A" + "BBA" * (n-3)) * 1

    fig = plt.figure(figsize = (3,10))

    ax = fig.add_subplot(1,2,1)
    data = run_simulation(word,n,np.pi/11)
    im = ax.imshow(data,origin = "lower",vmax= 0.7)
    plt.colorbar(im, ax = ax,fraction=0.05, pad=0.09)

    ax = fig.add_subplot(1,2,2)
    data = run_simulation(word,n,np.pi/7)
    im = ax.imshow(data,origin = "lower",vmax = 0.7)
    plt.colorbar(im, ax = ax,fraction=0.05, pad=0.09)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n = 15
    word = ("A" + "BBA" * (n-3)) * 1
    fid = []

    thetas = np.arange(0,np.pi/8,0.01)
    for th in range(len(thetas)):
        data = run_simulation(word,n,thetas[th])
        fid.append(data[-1,:].max())
        
    fig = plt.figure(figsize = (5,5))

    ax = fig.add_subplot(1,2,1)
    data = run_simulation(word,n,np.pi/11)
    im = ax.imshow(data,origin = "lower")
    ax.set_xlabel("position")
    ax.set_ylabel("time")
    plt.colorbar(im, ax = ax,fraction=0.05, pad=0.09)

    ax = fig.add_subplot(1,2,2)
    ax.plot(thetas,fid,color = "red")
    ax.plot(thetas,np.cos(thetas) **2,color = "blue")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__2":
    n = 4
    model = SynchronizingWalk(n,0)
    SS = model.Evo
    data = SS
    im = plt.imshow(data,origin = "upper")
    plt.colorbar(im, fraction=0.05, pad=0.09)
    plt.show()
    
    #print(np.roll(np.eye(5),-1,0))




