init_picnic = """ 
A 1s
8S
L 0.1s
0.5s
L_STICK@+000+100 1s
1s
A 1s
1s
A 1s
8s
"""

sandwich = """ 
LOOP 7
    L_STICK@+000-100 0.1s
    0.2s
A 0.1s
1s
A 0.1s
10s
L_STICK@+000+100 0.6s
1s
A L_STICK@+000-100 0.55s
2s
L_STICK@+000+100 0.6s
1s
A L_STICK@+000-100 0.6s
2s
L_STICK@+000+100 0.6s
1s
A L_STICK@+000-100 0.55s
2s
A 0.1s
3s
A 0.1s
3s
A 0.1s
10s
A 0.1s
2s
A 0.1s
25s
A 0.1s
1s
"""

box_walk = """ 
L 0.2s
0.5s
L_STICK@+000+100 0.5s
1s
L_STICK@-100+000 0.35s
1s
L_STICK@+000+100 0.4s
1s
L_STICK@+100+000 0.35s
1s
L 0.2s
0.5s
L_STICK@+100+000 0.1s
2s
"""


exit_picnic = """ 
Y 0.2s
1s
A 0.2s
10s
"""

mash_B_picnic = """ 
LOOP 3
    B 0.1s
    0.4s
"""

mash_B_hatch = """ 
LOOP 2
    B 0.1s
    1.9s
"""

mash_A_picnic = """ 
LOOP 30
    A 0.1s
    1.9s
"""




def load_eggs_from_col_i(i):
    macro_load_eggs = f""" 
LOOP {i+1}
    L_STICK@+100+000 0.1s
    0.2s
MINUS 0.1s
0.1s
LOOP 4
    L_STICK@+000-100 0.1s
    0.2s
A 0.1s
0.2s
LOOP {i+1}
    L_STICK@-100+000 0.1s
    0.2s
L_STICK@+000-100 0.1s
0.2s
A 0.1s
2s
L_STICK@+000+100 0.1s
0.2s
"""
    return(macro_load_eggs)

def return_hatchlings_to_col_i(i):
    macro_return_hatchlings = f""" 
L_STICK@+000-100 0.1s
0.2s
MINUS 0.1s
0.2s
LOOP 5
    L_STICK@+000-100 0.1s
    0.2s
A 0.1s
0.2s
LOOP {i+1}
    L_STICK@+100+000 0.1s
    0.2s
L_STICK@+000+100 0.1s
0.2s
A 0.1s
2s
"""
    return(macro_return_hatchlings)

macro_run_around = """ 
L_STICK@+000+100  R_STICK@-100+000 1s
"""
