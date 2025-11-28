import numpy as np
import matplotlib.pyplot as plt

m1 = 1
m2 = 2
l1 = 1.5
l2 = 1.5
g = 9.81
b1 = 2
b2 = 2


def dynamics(states:np.ndarray, 
             inputs = np.zeros((2,1)),
             friction:bool = True,
             complete:bool = True):
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

    global m1, m2, l1, l2, g, b1, b2

    o1 = states[0]
    o1_dot = states[1]
    o2 = states[2]
    o2_dot = states[3]


    M = np.array([
        [(m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(o2), m2 * l2**2 + m2 * l1 * l2 * np.cos(o2)],
        [m2 * l2**2 + m2 *l1 *l2 * np.cos(o2), m2 * l2**2]
    ])
    
    # OUR C
    C = np.array([
        [-m2 *l1*l2*np.sin(o2) * o2_dot, -m2 *l1*l2*np.sin(o2) * o2_dot], 
        [m2 * l1*l2*np.sin(o2) * o1_dot, m2 *l1*l2*np.sin(o2) * o1_dot]
    ])

    # C(q, q_dot) = c(q, q_dot)*q_dot
    C_complete = np.array([
        [-m2*l1*l2*np.sin(o2)*(2*o1_dot*o2_dot + o2_dot**2)],
		[m2*l1*l2*np.sin(o2)*o1_dot**2]
    ])


    if friction:
        D = np.array([
            [b1, 0],
            [0, b2]
        ])
    else:
        D = np.zeros((2, 2))

    # OUR G
    G = np.array([
        [(m1+m2)*g*l1*np.sin(o1) +  m2*g*l2*np.sin(o1 + o2)], 
        [m2*g*l2*np.sin(o1 + o2)]
    ])


    o_dot = np.array([
        [o1_dot], 
        [o2_dot]
        ])
    
    assert C_complete.any() == (C @ o_dot).any()

    if complete:
        o_dot_dot = - np.linalg.pinv(M) @ (C_complete + D @ o_dot + G + inputs)
    else:
        o_dot_dot = - np.linalg.pinv(M) @ (C @ o_dot + D @ o_dot + G + inputs)
    
    assert o_dot_dot.shape == (2, 1)

    o1_dot_dot = o_dot_dot[0, 0]
    assert isinstance(o1_dot_dot, float)
    o2_dot_dot = o_dot_dot[1, 0]
    assert isinstance(o2_dot_dot, float)

    
    states_dot = np.array([o1_dot, o1_dot_dot, o2_dot, o2_dot_dot])
    assert states_dot.shape == (4,)

    

    return states_dot


def solver(states, states_dot, dt, method:str = 'fe'):
    if method == 'rk':
        k1 = states_dot
        k2 = dynamics(states=states + dt * k1 / 2)
        k3 = dynamics(states=states + dt * k2 / 2)
        k4 = dynamics(states=states + dt * k3)
        
        new_states = states + dt/6 * (k1 + 2* k2 + 2 * k3 + k4)
    elif method == 'fe':
        new_states = states + dt * states_dot
    else:
        print("choose a valid method")
        exit()
    return new_states

def animation(states):
    steps = states.shape[0]
    
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()
    
    stop_animation = [False]  # Use list to allow modification in nested function
    
    def on_key(event):
        if event.key == 'q':
            stop_animation[0] = True
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    for i in range(steps):
        if stop_animation[0]:
            break
            
        ax.clear()
        o1 = states[i, 0]
        o2 = states[i, 2]
        
        p1 = np.array([l1*np.sin(o1), -l1*np.cos(o1)])
        p2 = np.array([l1*np.sin(o1) + l2*np.sin(o1+o2), -l1*np.cos(o1) -l2*np.cos(o1+o2)])

        lim_size = (l1+l2)*1.5

        ax.set_xlim([-lim_size, lim_size])
        ax.set_ylim([-lim_size, lim_size])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Double Pendulum - Step {i}')
        ax.plot([0.0, p1[0]], [0.0, p1[1]], 'grey', linewidth=2.5, label='Link 1')
        ax.scatter([0.0], [0.0], color='black', s=40, zorder=5)
        ax.scatter(p1[0], p1[1], color='blue', s=40, zorder=5)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', linewidth=2.5, label='Link 2')
        ax.scatter(p2[0], p2[1], color='red', s=40, zorder=5)
        ax.legend(loc='upper right')

        plt.draw()
        plt.pause(0.001)

def main():

    initial_state = np.array([np.pi/2, 0.0, np.pi/4, 0.0])
    
    steps = 1000
    states = np.zeros((steps, 4))

    states[0,:]= initial_state
    dt = 0.01

    for i in range(steps-1):
        states[i+1,:] = solver(states[i,:], dynamics(states=states[i,:]), dt)
    
    animation(states)
        

if __name__ == '__main__': 
    main()