import math

import torch

def get_W(n):

    W = [0, 0.22]

    while W[-1] < 1:
        W.append(W[-1] + max(0.06, W[-2] - W[-1] - 0.03))

    W = set(map(lambda x: math.ceil(x*n) if x < 1 else 0, W))
    return W

get_W(4)


def apgd(criterion_f, x0, projection_f, lr0, n_iter):
    W = get_W(n_iter)
    lr = lr0

    f0 = criterion_f(x0)
    f0.backward()
    g0 = x0.grad
    x = projection_f(x0 + lr0 * g0)
    f = criterion_f(x)
    if f0 > f:
        x_max = x0
        f_max = f0
    else:
        x_max = x
        f_max = f
    x_last = x0

    lr_changed = False
    improved = 0
    since_last_check = 0

    for iter in range(n_iter):
        since_last_check += 1

        f.backward()
        g = x.grad
        z_next = projection_f(x + lr * g)
        x_next = projection_f(x + 0.75 * (z_next - x) + 0.25 * (x - x_last))
        f_next = projection_f(x_next)

        if f_next > f_max:
            f_max = f_next
            x_max = x_next
            improved += 1

        x_last = x
        x = x_next

        if iter in W:
            condition1 = (improved / since_last_check) < 0.75
            condition2 = lr_changed and (improved > 0)
            if condition2 or condition1:
                lr /= 2
                lr_changed = True
                x = x_max
                since_last_check, improved = 0, 0
            else:
                lr_changed = False

        f = criterion_f(x)

print(get_W(4))








#%%
