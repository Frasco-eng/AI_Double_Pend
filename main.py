import numpy as np
import matplotlib.pyplot as plt

m1 = 1
m2 = 2
l1 = 1.5
l2 = 1.5
g = 9.81
b1 = 0.01
b2 = 0.03
dt = 0.01

def dynamics(states:np.ndarray, 
             inputs = np.zeros((2,1)),
             friction:bool = False):
    '''
    p_m1 = [l1 sin(o1), l1 cos(o1)]
    v_m1 = [l1 cos(o1) o1', -l1 sin(o1) o1']
    a_m1 = (-l1 sin(o1) o1'^2 + l1 cos(o1) o1'', -l1 cos(o1) o1'^2 - l1 sin(o1) o1'')

    p_m2 = [l1 sin(o1) + l2 sin (o1 + o2), l1 cos(o1) + l2 cos (o1 + o2)]
    v_m2  =[l1 cos(o1) o1' + l2 cos (o1 + o2) (o1' + o2'), -l1 sin(o1) o1' -l2 sin (o1 +  o2) (o1' + o2')]
    a_m2 = [-l1 sin(o1) o1'^2 + l1 cos(o1) o1'' - l2 sin(o1+o2) (o1' + o2' )^2 + l2 cos(o1 + o2) (o1'' + o2'')    , -l1 cos(o1) o1'^2 - l1 sin(o1) o1'' -l2 cos(o1 + o2)(o1' + o2')^2 - l2sin(o1 + o2)(o1'' + o2'')]
    

    
    (m1+m2) l1^2 o1'' + m2 l2^2 (o1'' + o2'') + m2 l1 l2 c2 (2o1'' + o2'') - m2 l1 l2 s2 (o1' + o2') o2' - (m1+m2) g l1 s1 - m2 g l2 s12 = 0
    l2^2 (o1'' + o2'') - m2 l1 l2 s2 o1' o2' + m2 l1 l2 c2 o1'' + m2 l1 l2 s2 o1' (o1' + o2') - m2 g l2 s12 = 0

    (m1 + m2) l1^2 o1'' + m2 l2^2 (o1'' + o2'') + m2 l1 l2 c2 (2o1'' + o2'') =  m2 l1 l2 s2 (o1' + o2') o2' + (m1+m2) g l1 s1 +  m2 g l2 s12
    l2^2 (o1'' + o2'') + m2 l1 l2 c2 o1'' = m2 l1 l2 s2 o1' o2' - m2 l1l2 s2 o1' (o1' + o2') + m2 g l2 s12

    
    # 1 
    ((m1 + m2) l1^2 + m2 l2^2 + 2 m2 l1 l2 c2)o1'' + (m2 l2^2 + m2 l1 l2 c2)o2'' - m2 l1 l2 s2 (o1' + o2') o2' - (m1+m2) g l1 s1 -  m2 g l2 s12 = 0
    
    # 2
    (l2^2 + m2 l1 l2 c2) o1'' + (l2^2) o2'' + m2 l1 l2 s2 o1' (o1' + o2') - m2 g l2 s12 = 0
    
    M o'' + C o'  + g = 0 
    
    o'' = - M^-1 (C o' + g)
    
    '''

    o1 = states[0]
    o1_dot = states[1]
    o2 = states[2]
    o2_dot = states[3]


    M = np.array([
        [(m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(o2), m2 * l2**2 + m2 * l1 * l2 * np.cos(o2)],
        [l2**2 + m2 *l1 *l2 * np.cos(o2), l2**2]
        ])
    

    C = np.array([
        [-m2 *l1*l2*np.sin(o2) * o2_dot, 0], 
        [0, -m2 * l1*l2*np.sin(o2) * o2_dot]
    ])

    if friction:
        D = np.array([
            [b1, 0],
            [0, b2]
        ])
    else:
        D = np.zeros((2, 2))


    g = np.array([
        [-(m1+m2)*g*l1*np.sin(o1) -  m2*g*l2*np.sin(o1 + o2)], 
        [-m2*g*l2*np.sin(o1 + o2)]
    ])


    o_dot = np.array([
        [o1_dot], 
        [o2_dot]
        ])
    
    # o'' = - M^-1 (C o' + D o' + g)
    o_dot_dot = - np.linalg.pinv(M) @ (C @ o_dot + D @ o_dot + g + inputs)

    o1_dot_dot = o_dot_dot[0]
    o2_dot_dot = o_dot_dot[1]


    states_dot = np.array([
        [o1_dot], 
        [o1_dot_dot], 
        [o2_dot], 
        [o2_dot_dot]
        ])


    return states_dot


def solver(states, states_dot, dt):
    return states + states_dot * dt


def main():

    actual_state = np.array([np.pi/2, 0, 0, 0])
    
    steps = 1000
    states = np.zeros((steps, 4))

    for i in range(steps):
        actual_state = (solver(actual_state, dynamics(actual_state)))
        states[i, :] = actual_state

    print(states)
        

if __name__ == '__main__': 
    main()