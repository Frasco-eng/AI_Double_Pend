import dynamics as dyn
import pandas as pd
import itertools


def main():

    # parameter grids
    params = {
        '01': list
        'm1': list(range(1, 10)),
        'm2': list(range(1, 10)),
        'l1': [round(1 + i * 0.1, 2) for i in range(0, 10)],
        'l2': [round(1 + i * 0.1, 2) for i in range(0, 10)],
        'g': [9.81],
        'b1': list(range(1, 5)),
        'b2': list(range(1, 5)),
    }

    keys = ['m1', 'm2', 'l1', 'l2', 'g', 'b1', 'b2']

    iterables = [params[k] for k in keys]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*iterables)]

    # create a dataframe with the columns: [ 'id', 'm1', 'm2', 'l1', 'l2', 'g', 'b1', 'b2', 'time_step', 'o1', 'o2', 'o1_dot', 'o2_dot' ]
    df = pd.DataFrame(columns=[
        'id_pendulum', 'm1', 'm2', 'l1', 'l2', 'g', 'b1', 'b2', 
        'time_step', 'o1', 'o2', 'o1_dot', 'o2_dot'
    ])

    dt = 0.01
    steps = 1000

    for i, combo in enumerate(combos):

        pendulum = dyn.DoublePendulum(
            m1=combo['m1'],
            m2=combo['m2'],
            l1=combo['l1'],
            l2=combo['l2'],
            g=combo['g'],
            b1=combo['b1'],
            b2=combo['b2']
        )

        for step in range(steps):

            
            
            o1, o2, o1_dot, o2_dot = states.flatten()

            df = pd.concat([df, pd.DataFrame([{
                'id_pendulum': i,
                'm1': combo['m1'],
                'm2': combo['m2'],
                'l1': combo['l1'],
                'l2': combo['l2'],
                'g': combo['g'],
                'b1': combo['b1'],
                'b2': combo['b2'],
                'time_step': step * dt,
                'o1': o1,
                'o2': o2,
                'o1_dot': o1_dot,
                'o2_dot': o2_dot
            }])], ignore_index=True)

            pendulum.step(dt)








if __name__ == '__main__':
    df = main()
    print(f'Generated {len(df)} combinations')



