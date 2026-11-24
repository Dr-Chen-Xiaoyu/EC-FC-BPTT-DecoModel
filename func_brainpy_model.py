'''
BrainPy Models written by Xiaoyu Chen(chenxy_sjtu@sjtu.edu.cn) and Yixiao Feng(newtonpula@sjtu.edu.cn) in 2023-11-03
'''
import brainpy as bp
import brainpy.math as bm
from jax import custom_jvp,vmap
from typing import Union,Callable
from brainpy.types import ArrayType


def Identity(x): # Identity function to do nothing
    return x


def get_AbbottChance(method = 'bm.ifelse', a=270, b=108, d=0.154, epsilon_inner=1e-1, epsilon_outer=5e2):
    ''' 
    AbbottChance refined using bm.ifelse for numerical stability and auto-grad :
        (-epsilon_inner, +epsilon_inner) section is replaced by a quadratic approximation 
        (-inf, -epsilon_outer) and (+epsilon_outer, +inf) sections are replaced by a ReLU approximation 
    '''
    def AbbottChance_ifelse(x):

        y = bm.ifelse( 
            operands=x, 
            conditions=(
                bm.abs(x)<epsilon_inner,
                bm.abs(x)<epsilon_outer,
                ),
            branches=(
                lambda x: d/12*x**2 + 0.5*x + 1/d, # case1 0点附近二次近似
                lambda x: x/(1 - bm.exp(-d*x)),    # case2 原始AbbottChance
                lambda x: bm.maximum(0,x),         # else  外围ReLU近似
                ),
        )
        
        return y


    ''' 
    AbbottChance refined using jax.custom_jvp with handcrafted gradiant for numerical stability and auto-grad :
        (-epsilon_inner, +epsilon_inner) section is replaced by a quadratic approximation 
    '''
    @custom_jvp
    def AbbottChance_customjvp(x):

        conditions = [
            (bm.abs(x) <= epsilon_inner),# section 1
            ((bm.abs(x) > epsilon_inner)&(bm.abs(x) < epsilon_outer)),# section 2
            (bm.abs(x) >= epsilon_outer),# section 3
        ]

        choices = [
            (d/12*x**2 + 0.5*x + 1/d),
            (x/(1 - bm.exp(-d*x))),
            (bm.maximum(0,x))
        ]

        return bm.select(conditions, choices)


    @AbbottChance_customjvp.defjvp
    def AbbottChance_customjvp_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        y = AbbottChance_customjvp(x)

        conditions = [
            (bm.abs(x) <= epsilon_inner),# section 1
            ((bm.abs(x) > epsilon_inner)&(bm.abs(x) < epsilon_outer)),# section 2
            (bm.abs(x) >= epsilon_outer),# section 3
        ]

        x1= bm.where( conditions[0], 
            x, #True
            epsilon_inner, #False,一个对所有情况都不会爆nan得安全的数即可
        )

        x2= bm.where( conditions[1], 
            x, #True
            epsilon_inner, #False
        )

        x3= bm.where( conditions[2], 
            x, #True
            epsilon_inner, #False
        )

        choices = [
            ( d/6*x1 + 0.5 ) * x_dot,
            ( 1 + (1 - d*x2)/(bm.exp(d*x2) - 1) - d*x2/(bm.exp(d*x2) - 1)**2 ) * x_dot,
            ( bm.heaviside(x3,0.5) ) * x_dot,
        ]

        y_dot = bm.select(conditions, choices)

        return y, y_dot
    
    
    # return the AbbottChance we want
    if method == 'bm.ifelse':
        AbbottChance_ifelse_m = vmap(vmap(AbbottChance_ifelse, out_axes=0,in_axes=0), out_axes=0,in_axes=0) # 2-D matrix element-wise operator
        def AbbottChance(x):
            return AbbottChance_ifelse_m(a*x-b)  
         
    elif method == 'custom_jvp':
        def AbbottChance(x):
            return AbbottChance_customjvp(a*x-b)
        
    else:
        raise ValueError("method from ['bm.ifelse','custom_jvp']")

    return AbbottChance




class DecoModel(bp.DynamicalSystemNS):
    """
    reduced-wong-wang-deco model using BrainPy 

    tau_S, w and I should be float or (1,) or (num,) bm.array, if float then it will be initialized as float*(num,) bm.array
    TrainVar_list should be a list such as ['tau_S','G','w','I'] to specify if this parameter is trainable
    
    S_init and H_init should be float or (num,) bm.array, if float then it will be initialized as float*(num,) bm.array
    S(gating state of model) and H(firing rate) will be initialized by S_init and H_init into (batch_size,num) with broadcasting along batch axis  

    """
    def __init__(
        self,
        size: int,
        struc_conn_matrix: ArrayType,
        batch_size: int = 1,
        gamma: float = 0.641, # kinetic parameter
        J: float = 0.2609, # synaptic coupling
        tau_S: Union[float, ArrayType] = 0.1, # time constant
        G: float = 1.0, # global coupling weight
        w: Union[float, ArrayType] = 0.9, # recurrent weights
        I: Union[float, ArrayType] = 0.0, # background inputs (intercepts)
        TrainVar_list = ['G','w','I'],
        H_x_act: Union[str, Callable] = 'Softplus', # callable element-wise activation function or just typing 'Softplus' or 'AbbottChance'
        S_init: Union[float, ArrayType] = None, # initial S
        H_init: Union[float, ArrayType] = None, # initial H (firing rate)
        bound_G: Callable = bm.abs, # function to bound/clip the parameters to a range, e.g., [0,inf)
        bound_w: Callable = Identity,
        bound_I: Callable = Identity,
    ):
        

        super(DecoModel, self).__init__()
        

        #>>> fixed parameters:
        self.num = size # number of network size (# of node)
        self.struc_conn_matrix = bm.asarray(struc_conn_matrix) # (num,num)
        self.gamma = gamma
        self.J = J

        #>>> function to bound/clip trainable weights
        self.bound_G = bound_G
        self.bound_w = bound_w
        self.bound_I = bound_I
        

        #>>> Trainable weights
        if 'tau_S' in TrainVar_list:
            if isinstance(tau_S, float):
                self.tau_S = bm.TrainVar(tau_S * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.tau_S = bm.TrainVar(tau_S) # (1,) or (num,)
        else:
            self.tau_S = tau_S # float or (1,) (num,) bm.array

        if 'G' in TrainVar_list:
            self.G = bm.TrainVar(G)
        else:
            self.G = G # float
        
        if 'w' in TrainVar_list:
            if isinstance(w, float):
                self.w = bm.TrainVar(w * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.w = bm.TrainVar(w) # (1,) or (num,)
        else:
            self.w = w # float or (1,) (num,) bm.array

        if 'I' in TrainVar_list:
            if isinstance(I, float):
                self.I = bm.TrainVar(I * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.I = bm.TrainVar(I) # (1,) or (num,)
        else:
            self.I = I # float or (1,) (num,) bm.array
        
            
        #>>> activation function
        if callable(H_x_act):
            self.H_x_act = H_x_act
        elif H_x_act == 'Softplus':
            self.H_x_act = lambda x: bm.logaddexp( 0.154*bm.log(2)*(270*x-108), 0 )/( 0.154*bm.log(2) ) # bp.dnn.Softplus(beta=0.154*bm.log(2))(270*x-108) 目前有错，更新后可再使用
        elif H_x_act == 'AbbottChance': # the original AbbottChance
            self.H_x_act = lambda x: bm.nan_to_num( (270*x-108)/(1-bm.exp(-0.154*(270*x-108))) , nan = 1/0.154 )


        #>>> Variables:
        if S_init is None:
            self.S_init = bm.zeros(self.num)
        elif isinstance(S_init, float):
            self.S_init = bm.asarray(S_init * bm.ones(self.num))
        else:
            self.S_init = bm.asarray(S_init) # (num,)
        

        if H_init is None:
            self.H_init = bm.zeros(self.num)
        elif isinstance(H_init, float):
            self.H_init = bm.asarray(H_init * bm.ones(self.num))
        else:
            self.H_init = bm.asarray(H_init) # (num,)
        

        self.S = bm.Variable(self.S_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H = bm.Variable(self.H_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
    
    
    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.S.value = self.S_init*bm.ones((batch_size,self.num))
        self.H.value = self.H_init*bm.ones((batch_size,self.num))
    

    def reset_init(self,):
        self.S_init.value = bm.mean(self.S,axis=0)
        self.H_init.value = bm.mean(self.H,axis=0)


    def update(self, inp = 0):
        # update S based on H and input, noise is integrated into the input
        self.S.value = self.S + ( - self.S / self.tau_S + self.gamma * (1-self.S) * self.H) * bm.dt + inp
        
        # hard sigmoid of S
        self.S.value = bm.minimum(bm.maximum(self.S, 0), 1)
        
        # get x based on S (modified 20241226, fix G to non-negative, add bound function for trainable paramters)
        x = self.J * bm.multiply(self.bound_w(self.w), self.S) + self.J * self.bound_G(self.G) * bm.matmul(self.S , self.struc_conn_matrix) + self.bound_I(self.I) 
        # x = self.J * bm.multiply(self.w, self.S) + self.J * bm.abs(self.G) * bm.matmul(self.S , self.struc_conn_matrix) + self.I 

        # get firing rate H(x) based on its input x
        self.H.value = self.H_x_act(x) 
        
        return self.S




class outLinear(bp.DynamicalSystemNS):
    '''
    Output-linear-scalling-layer of S from RNN-layer DecoModel
    size = num, i.e., number of network size (# of node)
    a or b must be float or bm.array with (1,) or (num,) shape
    if a or b is float, then set (num,) paramters with same initialization such as [a,a,a,a,....]
    if a or b is None, then it will not be included in scalling
    '''
    def __init__(
        self,
        size: int,
        a: Union[float, ArrayType] = None, # trainable
        b: Union[float, ArrayType] = None, # trainable
    ):

        super(outLinear, self).__init__()

        if a is None:
            self.a = 1.0
        elif isinstance(a,float):
            self.a = bm.TrainVar(a*bm.ones((size))) # (num,) with same initialization
        else:
            self.a = bm.TrainVar(a) # (1,) or (num,)

        if b is None:
            self.b = 0.0
        elif isinstance(b,float):
            self.b = bm.TrainVar(b*bm.ones((size))) # (num,) with same initialization
        else:
            self.b = bm.TrainVar(b) # (1,) or (num,)

    def update(self, inp = 0):
        return bm.multiply(self.a, inp) + self.b




class outBalloon(bp.DynamicalSystemNS):
    def __init__(
        self,
        size: int,
        batch_size: int = 1,
        rou: float = 0.34, # ρ = 0.34 is the resting oxygen extraction fraction, the p_constant
        tau: float = 0.98, # hemodynamic transit time τ = 0.98 s
        kappa: float = 0.65, # The kinetic parameters rate of signal decay κ
        gamma: float = 0.41, # rate of elimination
        alpha: float = 0.32, # Grubb’s exponent
        v_0: float = 0.02,# resting blood volume fraction
        B_0: float = 3.0, # B0, which is 3T in the HCP dataset
        r_0: float = 110, # the intravascular relaxation rate is r0 = 110 Hz
        e: float = 0.47, # the ratio between intravascular and extravascular MR signal is e = 0.47
        TE: float = 0.0331, # The echo time TE = 33.1 ms in the HCP dataset
        ):

        super(outBalloon, self).__init__()
        '''
        Output-nonlinear-scalling-layer (Balloon dynamics) of S from RNN-layer DecoModel, i.e., turning postsynaptic gating variable S to BOLD signal in an element-wise fashion.
        Balloon-Windkessel Hemodynamic model is used, equations with default parameters are from https://www.science.org/doi/10.1126/sciadv.aat7854
        
        size = num, i.e., number of network size (# of node)
        '''

        self.num = size

        self.rou = rou
        self.tau = tau
            
        self.kappa = kappa
        self.gamma = gamma
        self.alpha = alpha
 
        self.v_0 = v_0

        eta_0 = 28.265 * B_0 # ϑ0 = 28.265B0 is the frequency offset at the outer surface of magnetized vessels and depends on the main magnetic field strength B0
        self.k_1 = 4.3 * eta_0 * rou * TE
        self.k_2 = e * r_0 * rou * TE
        self.k_3 = 1-e
            
        self.F_0 = bm.Variable(0.0*bm.ones((batch_size,self.num)), batch_axis = 0)
        self.F_1 = bm.Variable(1.0*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.F_2 = bm.Variable(1.0*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.F_3 = bm.Variable(1.0*bm.ones((batch_size,self.num)), batch_axis = 0) 

    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.F_0.value = 0.0*bm.ones((batch_size,self.num)) 
        self.F_1.value = 1.0*bm.ones((batch_size,self.num)) 
        self.F_2.value = 1.0*bm.ones((batch_size,self.num)) 
        self.F_3.value = 1.0*bm.ones((batch_size,self.num)) 

    def update(self,S=0):
        '''
        input   S:  batch_size*size matrix represents synaptic gating variable
                batch_size is the number of batch
                size is the number of nodes

        output  S_BOLD: same shape matrix for BOLD signal output (element-wise Hemodynamic transformation)
        '''
                
        # gating2bold_derivative：
        dF_0 = S - self.kappa * self.F_0 - self.gamma * (self.F_1 -1)
        dF_1 = self.F_0
        dF_2 = 1 / self.tau * (self.F_1 - self.F_2**(1/self.alpha))
        dF_3 = 1 / self.tau * (self.F_1 / self.rou * (1-(1-self.rou)**(1/self.F_1)) - self.F_3 * self.F_2 **(1/self.alpha - 1))

        # forward-eular
        self.F_0.value = self.F_0 + dF_0*bm.dt
        self.F_1.value = self.F_1 + dF_1*bm.dt
        self.F_2.value = self.F_2 + dF_2*bm.dt
        self.F_3.value = self.F_3 + dF_3*bm.dt
        
        # caculate and return BOLD signal
        v_t = self.F_2
        q_t = self.F_3
        return self.v_0 * (self.k_1 * (1 - q_t) + self.k_2 * (1 - q_t / v_t) + self.k_3 * (1 - v_t))
        # return 100/self.rou*self.v_0 * (self.k_1 * (1 - q_t) + self.k_2 * (1 - q_t / v_t) + self.k_3 * (1 - v_t))
    



class DecoEIModel(bp.DynamicalSystemNS):
    """
    reduced-wong-wang-deco-like model with E(excitation) and I(inhibition) populations using BrainPy.
    
    Equations and default parameters are based on https://www.cell.com/neuron/fulltext/S0896-6273(19)30044-3 with same sytle of notations from original DecoModel.

    tau_S_E,tau_S_I, w_EE,w_EI,w_IE, I_E,I_I parameters should be float or (1,) or (num,) bm.array, if float then it will be initialized as float*(num,) bm.array.

    TrainVar_list should be a list such as ['tau_S','G','w_EE','I_E'] to specify if this parameter is trainable.
    
    S_E_init and H_E_init should be float or (num,) bm.array, if float then it will be initialized as float*(num,) bm.array. Same for inhibition S_I_init and H_I_init.
    
    S_E(gating state of model) and H_E(firing rate) will be initialized by S_E_init and H_I_init into (batch_size,num) with broadcasting along batch axis. Same for inhibition S_I and H_I.
    """
    def __init__(
        self,
        size: int,
        struc_conn_matrix: ArrayType,
        batch_size: int = 1,
        gamma: float = 0.641, # kinetic parameter
        J: float = 0.15, # synaptic coupling (nA)
        tau_S_E: Union[float, ArrayType] = 0.1,  # time constant of E (s)
        tau_S_I: Union[float, ArrayType] = 0.01, # time constant of I (s)
        G: float = 1.0, # global coupling weight, i.e., g in the paper
        w_EE: Union[float, ArrayType] = 1.0, # node recurrent weights E -> E
        w_EI: Union[float, ArrayType] = 1.0, # node recurrent weights E -> I
        w_IE: Union[float, ArrayType] = 1.0, # node recurrent weights I -> E
        I_E: Union[float, ArrayType] = 1.0*0.382, # background inputs (intercepts) of E (nA), i.e., w_E*I_b in the paper
        I_I: Union[float, ArrayType] = 0.7*0.382, # background inputs (intercepts) of I (nA), i.e., w_I*I_b in the paper
        TrainVar_list = ['G','w_EE','w_EI','w_IE','I_E','I_I'],
        H_E_x_act: Union[str, Callable] = get_AbbottChance(method='custom_jvp',a=310, b=125, d=0.160,), 
        H_I_x_act: Union[str, Callable] = get_AbbottChance(method='custom_jvp',a=615, b=177, d=0.087,), 
        # 2D element-wise AbbottChance function, it can be some other callable activation function or just typing 'Softplus' or 'AbbottChance'
        S_E_init: Union[float, ArrayType] = None, # initial S_E (gating state of E)
        H_E_init: Union[float, ArrayType] = None, # initial H_E (firing rate of E), i.e., r_E in the paper
        S_I_init: Union[float, ArrayType] = None, # initial S_I (gating state of I)
        H_I_init: Union[float, ArrayType] = None, # initial H_I (firing rate of I), i.e., r_I in the paper
    ):
        

        super(DecoEIModel, self).__init__()
        

        #>>> fixed parameters:
        self.num = size # number of network size (# of node)
        self.struc_conn_matrix = bm.asarray(struc_conn_matrix) # (num,num)
        self.gamma = gamma
        self.J = J
        

        #>>> Trainable weights
        if 'tau_S_E' in TrainVar_list:
            if isinstance(tau_S_E, float):
                self.tau_S_E = bm.TrainVar(tau_S_E * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.tau_S_E = bm.TrainVar(tau_S_E) # (1,) or (num,)
        else:
            self.tau_S_E = tau_S_E # float or (1,) (num,) bm.array

        if 'tau_S_I' in TrainVar_list:
            if isinstance(tau_S_I, float):
                self.tau_S_I = bm.TrainVar(tau_S_I * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.tau_S_I = bm.TrainVar(tau_S_I) # (1,) or (num,)
        else:
            self.tau_S_I = tau_S_I # float or (1,) (num,) bm.array

        if 'G' in TrainVar_list:
            self.G = bm.TrainVar(G)
        else:
            self.G = G # float
        
        if 'w_EE' in TrainVar_list:
            if isinstance(w_EE, float):
                self.w_EE = bm.TrainVar(w_EE * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.w_EE = bm.TrainVar(w_EE) # (1,) or (num,)
        else:
            self.w_EE = w_EE # float or (1,) (num,) bm.array

        if 'w_EI' in TrainVar_list:
            if isinstance(w_EI, float):
                self.w_EI = bm.TrainVar(w_EI * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.w_EI = bm.TrainVar(w_EI) # (1,) or (num,)
        else:
            self.w_EI = w_EI # float or (1,) (num,) bm.array

        if 'w_IE' in TrainVar_list:
            if isinstance(w_IE, float):
                self.w_IE = bm.TrainVar(w_IE * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.w_IE = bm.TrainVar(w_IE) # (1,) or (num,)
        else:
            self.w_IE = w_IE # float or (1,) (num,) bm.array

        if 'I_E' in TrainVar_list:
            if isinstance(I_E, float):
                self.I_E = bm.TrainVar(I_E * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.I_E = bm.TrainVar(I_E) # (1,) or (num,)
        else:
            self.I_E = I_E # float or (1,) (num,) bm.array

        if 'I_I' in TrainVar_list:
            if isinstance(I_I, float):
                self.I_I = bm.TrainVar(I_I * bm.ones(self.num)) # (num,) with same initialization
            else:
                self.I_I = bm.TrainVar(I_I) # (1,) or (num,)
        else:
            self.I_I = I_I # float or (1,) (num,) bm.array
        
            
        #>>> activation function
        if callable(H_E_x_act):
            self.H_E_x_act = H_E_x_act
        elif H_E_x_act == 'Softplus':
            self.H_E_x_act = lambda x: bm.logaddexp( 0.154*bm.log(2)*(270*x-108), 0 )/( 0.154*bm.log(2) ) # bp.dnn.Softplus(beta=0.154*bm.log(2))(270*x-108) 目前有错，更新后可再使用
        elif H_E_x_act == 'AbbottChance': # the original AbbottChance
            self.H_E_x_act = lambda x: bm.nan_to_num( (270*x-108)/(1-bm.exp(-0.154*(270*x-108))) , nan = 1/0.154 )

        if callable(H_I_x_act):
            self.H_I_x_act = H_I_x_act
        elif H_I_x_act == 'Softplus':
            self.H_I_x_act = lambda x: bm.logaddexp( 0.154*bm.log(2)*(270*x-108), 0 )/( 0.154*bm.log(2) ) # bp.dnn.Softplus(beta=0.154*bm.log(2))(270*x-108) 目前有错，更新后可再使用
        elif H_I_x_act == 'AbbottChance': # the original AbbottChance
            self.H_I_x_act = lambda x: bm.nan_to_num( (270*x-108)/(1-bm.exp(-0.154*(270*x-108))) , nan = 1/0.154 )


        #>>> Variables:
        if S_E_init is None:
            self.S_E_init = bm.zeros(self.num)
        elif isinstance(S_E_init, float):
            self.S_E_init = bm.asarray(S_E_init * bm.ones(self.num))
        else:
            self.S_E_init = bm.asarray(S_E_init) # (num,)
        
        if S_I_init is None:
            self.S_I_init = bm.zeros(self.num)
        elif isinstance(S_I_init, float):
            self.S_I_init = bm.asarray(S_I_init * bm.ones(self.num))
        else:
            self.S_I_init = bm.asarray(S_I_init) # (num,)

        if H_E_init is None:
            self.H_E_init = bm.zeros(self.num)
        elif isinstance(H_E_init, float):
            self.H_E_init = bm.asarray(H_E_init * bm.ones(self.num))
        else:
            self.H_E_init = bm.asarray(H_E_init) # (num,)
        
        if H_I_init is None:
            self.H_I_init = bm.zeros(self.num)
        elif isinstance(H_I_init, float):
            self.H_I_init = bm.asarray(H_I_init * bm.ones(self.num))
        else:
            self.H_I_init = bm.asarray(H_I_init) # (num,)

        self.S_E = bm.Variable(self.S_E_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H_E = bm.Variable(self.H_E_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
    
        self.S_I = bm.Variable(self.S_I_init*bm.ones((batch_size,self.num)), batch_axis = 0) 
        self.H_I = bm.Variable(self.H_I_init*bm.ones((batch_size,self.num)), batch_axis = 0) 

    def reset_state(self, batch_size=1): # this function defines how to reset the mode states
        self.S_E.value = self.S_E_init*bm.ones((batch_size,self.num))
        self.H_E.value = self.H_E_init*bm.ones((batch_size,self.num))
    
        self.S_I.value = self.S_I_init*bm.ones((batch_size,self.num))
        self.H_I.value = self.H_I_init*bm.ones((batch_size,self.num))

    def reset_init(self,):
        self.S_E_init.value = bm.mean(self.S_E,axis=0)
        self.H_E_init.value = bm.mean(self.H_E,axis=0)

        self.S_I_init.value = bm.mean(self.S_I,axis=0)
        self.H_I_init.value = bm.mean(self.H_I,axis=0)


    def update(self, inp = 0):
        # update S based on H and input, noise is integrated into the input
        dS_E = - self.S_E / self.tau_S_E + self.gamma * (1-self.S_E) * self.H_E
        dS_I = - self.S_I / self.tau_S_I + self.H_I

        self.S_E.value = self.S_E + dS_E * bm.dt + inp
        self.S_I.value = self.S_I + dS_I * bm.dt
        
        # hard sigmoid of S
        self.S_E.value = bm.minimum(bm.maximum(self.S_E, 0), 1)
        self.S_I.value = bm.minimum(bm.maximum(self.S_I, 0), 1)
        
        # get x, i.e., current I in the paper, based on S
        x_E = bm.abs(self.I_E) + self.J * bm.abs(self.w_EE) * self.S_E - self.J * bm.abs(self.w_IE) * self.S_I + self.J * bm.abs(self.G) * bm.matmul(self.S_E , self.struc_conn_matrix)
        x_I = bm.abs(self.I_I) + self.J * bm.abs(self.w_EI) * self.S_E - self.S_I
        
        # get firing rate H(x) based on its input x
        self.H_E.value = self.H_E_x_act(x_E) 
        self.H_I.value = self.H_I_x_act(x_I) 
        
        return self.S_E